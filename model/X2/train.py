import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings('ignore')

# =======================================
# DATA LOADING AND EXPLORATION
# =======================================

print("Loading data...")
df = pd.read_csv('nse_companies.csv')

print(f"Total companies: {len(df)}")
print(f"Dataset columns: {df.columns.tolist()}")
print("\nSample data (5 rows):")
print(df.head())

# Replace 'Unknown' with NaN for clarity
df['intrinsic_label'] = df['intrinsic_label'].replace('Unknown', np.nan)

# =======================================
# COMPREHENSIVE FEATURE ENGINEERING
# =======================================

print("\nPerforming comprehensive feature engineering...")

# 1. Profitability Ratios
df['profit_margin'] = df['avg_profit_loss_for_period'] / df['avg_revenue_from_operations'].replace(0, np.nan)
df['gross_profit_margin'] = (df['avg_revenue_from_operations'] - df['avg_other_expenses']) / df['avg_revenue_from_operations'].replace(0, np.nan)
df['return_on_equity'] = df['avg_profit_loss_for_period'] / df['avg_paid_up_equity_share_capital'].replace(0, np.nan)
df['return_on_assets'] = df['avg_profit_loss_for_period'] / (df['avg_revenue_from_operations'] + df['avg_other_income']).replace(0, np.nan)
df['operating_margin'] = df['avg_profit_before_tax'] / df['avg_revenue_from_operations'].replace(0, np.nan)

# 2. Efficiency Ratios
df['asset_turnover'] = df['avg_revenue_from_operations'] / (df['avg_revenue_from_operations'] + df['avg_other_income']).replace(0, np.nan)
df['operational_efficiency'] = (df['avg_employee_benefit_expense'] + df['avg_other_expenses']) / df['avg_revenue_from_operations'].replace(0, np.nan)
df['employee_productivity'] = df['avg_revenue_from_operations'] / df['avg_employee_benefit_expense'].replace(0, np.nan)

# 3. Leverage and Solvency Ratios
df['debt_to_income'] = df['avg_finance_costs'] / df['avg_income'].replace(0, np.nan)
df['interest_coverage'] = df['avg_profit_before_tax'] / df['avg_finance_costs'].replace(0, np.nan)

# 4. Valuation Metrics
df['earnings_growth'] = df['avg_comprehensive_income'] / df['avg_paid_up_equity_share_capital'].replace(0, np.nan)
df['price_to_earnings'] = df['avg_close'] / df['avg_basic_earnings_per_share'].replace(0, np.nan)
df['price_to_book'] = df['avg_close'] / (df['avg_paid_up_equity_share_capital'] / df['issued_size']).replace(0, np.nan)

# 5. Market Performance Metrics
df['volume_price_ratio'] = df['avg_volume'] / df['avg_close'].replace(0, np.nan)
df['high_low_ratio'] = df['avg_high'] / df['avg_low'].replace(0, np.nan)
df['price_volatility'] = df['volatility'] / df['avg_close'].replace(0, np.nan)

# 6. Transform market_cap_category to numeric
market_cap_mapping = {'Large': 4, 'Mid': 3, 'Small': 2, 'Micro': 1, np.nan: 0}
df['market_cap_numeric'] = df['market_cap_category'].map(market_cap_mapping)

# 7. Size metrics with log transformations to handle skewness
df['log_revenue'] = np.log1p(df['avg_revenue_from_operations'])
df['log_profit'] = np.log1p(df['avg_profit_loss_for_period'].clip(lower=0))
df['log_market_cap'] = df['market_cap_numeric'] * df['log_revenue'] / 5

# 8. Combined metrics
df['financial_health'] = (
    df['profit_margin'].fillna(0) + 
    df['return_on_equity'].fillna(0) + 
    df['sharpe_ratio'].fillna(0) -
    df['volatility'].fillna(0) / 100
)

# 9. Piotroski F-Score Components 
# Simplified implementation of key components
df['positive_profit'] = (df['avg_profit_loss_for_period'] > 0).astype(int)
df['positive_operating_cash_flow'] = (df['avg_profit_before_tax'] > 0).astype(int)
df['decreasing_debt'] = ((df['avg_finance_costs'] / df['avg_income']) < 0.5).astype(int)
df['improving_margin'] = (df['profit_margin'] > df['profit_margin'].median()).astype(int)

# 10. Altman Z-Score Components (simplified)
df['working_capital_to_assets'] = (df['avg_profit_before_tax'] / df['avg_revenue_from_operations']).replace(0, np.nan)
df['retained_earnings_to_assets'] = (df['avg_comprehensive_income'] / df['avg_revenue_from_operations']).replace(0, np.nan)

# =======================================
# CREATE TARGET VARIABLE
# =======================================

print("\nCreating target variable for intrinsic strength...")

# Method 1: Multi-factor ranking approach
# Combine multiple financial health indicators
df['strength_score'] = (
    df['return_on_equity'].rank(pct=True) + 
    df['profit_margin'].rank(pct=True) + 
    df['sharpe_ratio'].rank(pct=True) + 
    df['financial_health'].rank(pct=True) - 
    df['volatility'].rank(pct=True) / 2   # Lower volatility is better
) / 4.5  # Normalize to roughly 0-1 range

# Define 'strong' stocks as those in the top 20% of strength scores
strength_threshold = df['strength_score'].quantile(0.80)
df['is_strong'] = (df['strength_score'] > strength_threshold).astype(int)

# Alternative Method 2: Use existing labels as guidance
# Map intrinsic labels to strength values (Large and Mid caps are generally stronger)
label_strength = {
    'IN-L': 1,  # Nifty Large cap = strong
    'IN-M': 1,  # Nifty Mid cap = strong
    'IN-S': 1,  # Nifty Small cap = strong
    'IN-Mi': 1  # Nifty Micro cap = strong
}

# For stocks with labels, use them to set strength
labeled_mask = df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])
df.loc[labeled_mask, 'is_strong'] = 1

print(f"Created {df['is_strong'].sum()} strong stocks out of {len(df)} total stocks")
print(f"Average strength score: {df['strength_score'].mean():.4f}")
print(f"Using threshold: {strength_threshold:.4f}")

# =======================================
# FEATURE SELECTION
# =======================================

print("\nPerforming feature selection...")

# Define candidate features
features = [
    # Profitability
    'profit_margin', 'gross_profit_margin', 'return_on_equity', 'return_on_assets', 'operating_margin',
    
    # Efficiency
    'asset_turnover', 'operational_efficiency', 'employee_productivity',
    
    # Leverage
    'debt_to_income', 'interest_coverage',
    
    # Valuation
    'earnings_growth', 'price_to_earnings', 'price_to_book',
    
    # Market metrics
    'market_cap_numeric', 'volume_price_ratio', 'high_low_ratio', 'price_volatility',
    'volatility', 'sharpe_ratio', 'avg_daily_return', 'avg_vwap_distance',
    'yearly_momentum_score', 'liquidity_score', 'stability_score', 'trend_strength',
    
    # Basic financials
    'avg_basic_earnings_per_share', 'avg_diluted_earnings_per_share',
    
    # Log-transformed size metrics
    'log_revenue', 'log_profit',
    
    # F-Score and Z-Score components
    'positive_profit', 'positive_operating_cash_flow', 'decreasing_debt', 'improving_margin',
    'working_capital_to_assets', 'retained_earnings_to_assets',
    
    # Combined metrics
    'financial_health'
]

# Handle missing values with median imputation
print("Handling missing and infinite values...")
for feature in features:
    if feature in df.columns:
        # Replace infinities
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        # Replace NaNs with median
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)

# Check feature correlation with target (is_strong)
correlation_with_target = []
for feature in features:
    if feature in df.columns:
        corr = df[feature].corr(df['is_strong'])
        if not np.isnan(corr):
            correlation_with_target.append((feature, corr))

# Sort by absolute correlation
correlation_with_target.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nTop 15 features correlated with intrinsic strength:")
for feature, corr in correlation_with_target[:15]:
    print(f"{feature}: {corr:.4f}")

# Select top features based on correlation
top_n_features = 25  # Adjust based on feature importance later
selected_features = [f[0] for f in correlation_with_target[:top_n_features]]

print(f"\nSelected {len(selected_features)} features for modeling:")
for feature in selected_features:
    print(f"- {feature}")

# =======================================
# DATA PREPARATION
# =======================================

print("\nPreparing data for XGBoost modeling...")

# Prepare X and y
X = df[selected_features]
y = df['is_strong']

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

# Split data - using stratified sampling to keep class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Positive class distribution - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")

# =======================================
# XGBOOST MODEL TRAINING
# =======================================

print("\nTraining XGBoost model with grid search...")

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3]
}

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

# Use TimeSeriesSplit for financial data (simulates trading strategy validation)
tscv = TimeSeriesSplit(n_splits=5)

# Grid search with cross-validation
grid_search = GridSearchCV(
    xgb_model,
    param_grid,
    cv=5,  # Using 5-fold CV instead of tscv since we don't have true time series data
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Get the best model
best_model = grid_search.best_estimator_

# =======================================
# MODEL EVALUATION
# =======================================

print("\nEvaluating model on test set...")

# Make predictions
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
try:
    roc_auc = roc_auc_score(y_test, y_pred_proba)
except:
    roc_auc = 0

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')
print("ROC curve saved as 'roc_curve.png'")

# Precision-Recall curve
plt.figure(figsize=(8, 6))
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
plt.axhline(y=sum(y_test)/len(y_test), color='gray', lw=1, linestyle='--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig('precision_recall_curve.png')
print("Precision-Recall curve saved as 'precision_recall_curve.png'")

# =======================================
# FEATURE IMPORTANCE ANALYSIS
# =======================================

print("\nAnalyzing feature importance...")

# Get feature importance from XGBoost
feature_importance = best_model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.bar(range(len(indices)), feature_importance[indices])
plt.xticks(range(len(indices)), [selected_features[i] for i in indices], rotation=90)
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance saved as 'feature_importance.png'")

# Print feature importance ranking
print("\nFeature importance ranking:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {selected_features[idx]}: {feature_importance[idx]:.4f}")

# Try to use SHAP for more detailed feature importance
try:
    print("\nCalculating SHAP values...")
    explainer = shap.Explainer(best_model)
    shap_values = explainer(X_test)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    print("SHAP importance saved as 'shap_importance.png'")
    
    # Individual SHAP explanations for top predictions
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig('shap_explanation.png')
    print("SHAP explanation saved as 'shap_explanation.png'")
except Exception as e:
    print(f"SHAP analysis failed: {e}")

# =======================================
# IDENTIFY TOP INTRINSICALLY STRONG STOCKS
# =======================================

print("\nIdentifying top intrinsically strong stocks...")

# Make predictions on the entire dataset
X_all_scaled = scaler.transform(df[selected_features])
df['strength_probability'] = best_model.predict_proba(X_all_scaled)[:, 1]

# MODIFICATION: Filter out companies that are already prelabelled (IN- series)
# Create a mask for companies without IN- labels
unlabeled_mask = ~df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])
non_prelabeled_df = df[unlabeled_mask]

# Sort by predicted probability among non-prelabeled stocks
top_stocks = non_prelabeled_df.sort_values('strength_probability', ascending=False).head(50)

print("\n===== TOP 20 INTRINSICALLY STRONG STOCKS (EXCLUDING PRELABELED STOCKS) =====")
for i in range(min(20, len(top_stocks))):
    row = top_stocks.iloc[i]
    print(f"{i+1}. {row['symbol']} - {row['company_name']} (Prob: {row['strength_probability']:.4f})")

# Save stats on how many of each category we found
print("\n===== COMPARISON WITH PRELABELED STOCKS =====")
total_df_high_prob = df[df['strength_probability'] > 0.8]
prelabeled_high_prob = total_df_high_prob[~unlabeled_mask]
print(f"Total high probability (>0.8) stocks found: {len(total_df_high_prob)}")
print(f"Prelabeled high probability stocks: {len(prelabeled_high_prob)} ({len(prelabeled_high_prob)/len(total_df_high_prob)*100:.1f}%)")
print(f"New discoveries high probability stocks: {len(total_df_high_prob) - len(prelabeled_high_prob)} ({(len(total_df_high_prob) - len(prelabeled_high_prob))/len(total_df_high_prob)*100:.1f}%)")

# Save top non-prelabeled stocks to CSV
top_stocks[['symbol', 'company_name', 'company_industry', 'sector', 'strength_probability', 
           'market_cap_category', 'intrinsic_label', 'sharpe_ratio', 'return_on_equity',
           'profit_margin', 'year_change_percent']].to_csv('top_strong_stocks_non_prelabeled.csv', index=False)
print("\nSaved top strong non-prelabeled stocks to 'top_strong_stocks_non_prelabeled.csv'")

# =======================================
# GENERATE DETAILED REPORT
# =======================================

print("\nGenerating detailed report...")

with open('stock_strength_model_report.txt', 'w') as f:
    f.write("STOCK INTRINSIC STRENGTH CLASSIFICATION REPORT\n")
    f.write("===========================================\n\n")
    
    f.write("DATA SUMMARY\n")
    f.write("-----------\n")
    f.write(f"Total companies analyzed: {len(df)}\n")
    f.write(f"Companies classified as intrinsically strong: {df['is_strong'].sum()} ({df['is_strong'].mean()*100:.1f}%)\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("----------------\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
    f.write(f"PR AUC: {pr_auc:.4f}\n\n")
    
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n")
    
    f.write("FEATURE IMPORTANCE\n")
    f.write("----------------\n")
    for i, idx in enumerate(indices):
        f.write(f"{i+1}. {selected_features[idx]}: {feature_importance[idx]:.4f}\n")
    f.write("\n")
    
    f.write("TOP INTRINSICALLY STRONG STOCKS (EXCLUDING PRELABELED STOCKS)\n")
    f.write("------------------------------------------------------\n")
    f.write("Top 50 non-prelabeled stocks by predicted strength probability:\n\n")
    
    for i in range(min(50, len(top_stocks))):
        row = top_stocks.iloc[i]
        f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
        f.write(f"   Strength Probability: {row['strength_probability']:.4f}\n")
        f.write(f"   Market Cap: {row['market_cap_category']}\n")
        f.write(f"   Industry: {row['company_industry']}\n")
        f.write(f"   Sector: {row['sector']}\n")
        f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
        f.write(f"   Return on Equity: {row['return_on_equity']:.4f}\n")
        f.write(f"   Profit Margin: {row['profit_margin']:.4f}\n")
        f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n\n")
    
    f.write("\nCOMPARISON WITH PRELABELED STOCKS\n")
    f.write("------------------------------\n")
    f.write(f"Total high probability (>0.8) stocks found: {len(total_df_high_prob)}\n")
    f.write(f"Prelabeled high probability stocks: {len(prelabeled_high_prob)} ({len(prelabeled_high_prob)/len(total_df_high_prob)*100:.1f}%)\n")
    f.write(f"New discoveries high probability stocks: {len(total_df_high_prob) - len(prelabeled_high_prob)} ({(len(total_df_high_prob) - len(prelabeled_high_prob))/len(total_df_high_prob)*100:.1f}%)\n")

print("\nAnalysis complete! You now have:")
print("1. An XGBoost model for predicting intrinsically strong stocks")
print("2. Multiple visualizations showing model performance and feature importance")
print("3. A list of the top 50 predicted strong stocks (excluding prelabeled stocks)")
print("4. A detailed report with model evaluation and stock recommendations")
print("5. A comparison between prelabeled and newly discovered strong stocks")