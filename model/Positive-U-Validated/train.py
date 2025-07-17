import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve, auc, roc_curve, roc_auc_score
import xgboost as xgb
import time
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

# Check distribution of labels
print("\nDistribution of intrinsic labels:")
print(df['intrinsic_label'].value_counts(dropna=False))

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

# =======================================
# PU LEARNING IMPLEMENTATION
# =======================================

print("\nImplementing Positive-Unlabeled Learning for stock classification...")

# Step 1: Identify strong (positive) and unknown (unlabeled/validation) samples
print("Identifying strong-labeled (positive) and NaN-labeled (unlabeled) samples...")

STRONG_LABELS = {'IN-L', 'IN-M', 'IN-S', 'IN-Mi'}
positive_df = df[df['intrinsic_label'].isin(STRONG_LABELS)].copy()
unlabeled_df = df[df['intrinsic_label'].isna()].copy()

print(f"Positive (strong) samples: {len(positive_df)}")
print(f"Unlabeled (validation) samples: {len(unlabeled_df)}")

# Step 2: Select features and prepare data
print("\nPreparing feature data...")
selected_features = features.copy()  # Using all features initially

# Scale features
print("Scaling features...")
scaler = RobustScaler()
X_positive = positive_df[selected_features]
X_positive_scaled = scaler.fit_transform(X_positive)

X_unlabeled = unlabeled_df[selected_features]
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Step 3: Set up Bagging-based PU Learning
print("\nImplementing Bagging-based PU Learning...")

# Create base XGBoost classifier
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'min_child_weight': 3,
    'n_estimators': 100,
    'random_state': 42
}

base_classifier = xgb.XGBClassifier(**base_params)

# Bagging parameters
n_bootstraps = 50  # Number of bootstrap iterations
bootstrap_predictions = np.zeros((len(unlabeled_df), n_bootstraps))
positives_per_bootstrap = len(positive_df)
unlabeled_per_bootstrap = positives_per_bootstrap  # Use equal sampling

# Step 4: Feature importance aggregation for tracking
feature_importances = np.zeros(len(selected_features))

print(f"Running {n_bootstraps} bootstrap iterations...")
start_time = time.time()

# Step 5: Perform bagging iterations
for i in range(n_bootstraps):
    if i % 10 == 0:
        print(f"  Bootstrap iteration {i+1}/{n_bootstraps}...")
    
    # Use all positive examples
    sampled_positives = positive_df.copy()
    sampled_positives['label'] = 1
    
    # Sample from unlabeled examples
    sampled_unlabeled = unlabeled_df.sample(n=unlabeled_per_bootstrap, replace=True, random_state=i)
    sampled_unlabeled['label'] = 0
    
    # Combine datasets
    bootstrap_df = pd.concat([sampled_positives, sampled_unlabeled])
    X_boot = bootstrap_df[selected_features]
    y_boot = bootstrap_df['label']
    X_boot_scaled = scaler.transform(X_boot)
    
    # Train model
    bootstrap_model = xgb.XGBClassifier(**base_params)
    bootstrap_model.fit(X_boot_scaled, y_boot)
    
    # Store feature importances
    feature_importances += bootstrap_model.feature_importances_
    
    # Predict on all unlabeled data
    bootstrap_predictions[:, i] = bootstrap_model.predict_proba(X_unlabeled_scaled)[:, 1]

# Calculate average feature importance
feature_importances /= n_bootstraps

print(f"Bagging completed in {time.time() - start_time:.1f} seconds")

# Step 6: Calculate aggregate scores and confidence metrics
print("\nCalculating aggregate prediction scores and confidence metrics...")

# Calculate raw prediction statistics
unlabeled_df['mean_probability'] = np.mean(bootstrap_predictions, axis=1)
unlabeled_df['std_probability'] = np.std(bootstrap_predictions, axis=1)
unlabeled_df['min_probability'] = np.min(bootstrap_predictions, axis=1)
unlabeled_df['max_probability'] = np.max(bootstrap_predictions, axis=1)
unlabeled_df['median_probability'] = np.median(bootstrap_predictions, axis=1)

# Calculate confidence metrics
unlabeled_df['confidence_score'] = unlabeled_df['mean_probability']
unlabeled_df['uncertainty'] = unlabeled_df['std_probability']
unlabeled_df['prediction_range'] = unlabeled_df['max_probability'] - unlabeled_df['min_probability']

# Step 7: Apply thresholds to assign final labels
threshold = 0.4243  # Can be adjusted based on validation if available
unlabeled_df['is_strong'] = (unlabeled_df['mean_probability'] > threshold).astype(int)
unlabeled_df['high_confidence'] = (unlabeled_df['confidence_score'] > 0.75) & (unlabeled_df['uncertainty'] < 0.15)

# Create a final prediction dataframe with relevant stock information
final_predictions = unlabeled_df.copy()
final_predictions['predicted_label'] = np.where(final_predictions['is_strong'] == 1, 'Strong', 'Not Strong')

# =======================================
# VALIDATION SETUP 
# =======================================

print("\nSetting up validation framework...")

# Hold out 20% of positive samples for validation
from sklearn.model_selection import train_test_split
positive_train_df, positive_validation_df = train_test_split(
    positive_df, test_size=0.2, random_state=42)

print(f"Positive training samples: {len(positive_train_df)}")
print(f"Positive validation samples: {len(positive_validation_df)}")

# We'll now use positive_train_df instead of positive_df in the PU learning steps

# Keep a copy of the original bootstrap process to use with validation later
def train_bootstrap_model(pos_df, unl_df, iter_idx):
    """Helper function to train a bootstrap model with consistent parameters"""
    # Use all positive examples
    sampled_positives = pos_df.copy()
    sampled_positives['label'] = 1
    
    # Sample from unlabeled examples
    sampled_unlabeled = unl_df.sample(n=len(sampled_positives), replace=True, random_state=iter_idx)
    sampled_unlabeled['label'] = 0
    
    # Combine datasets
    bootstrap_df = pd.concat([sampled_positives, sampled_unlabeled])
    X_boot = bootstrap_df[selected_features]
    y_boot = bootstrap_df['label']
    X_boot_scaled = scaler.transform(X_boot)
    
    # Train model
    bootstrap_model = xgb.XGBClassifier(**base_params)
    bootstrap_model.fit(X_boot_scaled, y_boot)
    
    return bootstrap_model

# =======================================
# CROSS-VALIDATION ON POSITIVE SAMPLES
# =======================================

print("\nPerforming 5-fold cross-validation on known positive samples...")

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []
cv_aucs = []
cv_precision = []
cv_recall = []

cv_bootstrap_iterations = 10  # Use fewer bootstraps for CV to save time

for fold, (train_idx, val_idx) in enumerate(kf.split(positive_df)):
    print(f"  Processing fold {fold+1}/5...")
    
    # Split positive data into training and validation
    pos_train = positive_df.iloc[train_idx]
    pos_val = positive_df.iloc[val_idx]
    
    # Run bagging PU learning with current fold's positive training data
    fold_predictions = np.zeros((len(pos_val), cv_bootstrap_iterations))
    unlabeled_fold_preds = np.zeros((len(unlabeled_df.sample(len(pos_val))), cv_bootstrap_iterations))
    
    for i in range(cv_bootstrap_iterations):
        bootstrap_model = train_bootstrap_model(pos_train, unlabeled_df, i*100+fold)
        
        # Predict on validation positives
        X_val = pos_val[selected_features]
        X_val_scaled = scaler.transform(X_val)
        fold_predictions[:, i] = bootstrap_model.predict_proba(X_val_scaled)[:, 1]
        
        # Predict on random unlabeled samples (as proxy for negatives)
        X_unl_sample = unlabeled_df.sample(len(pos_val), random_state=i*100+fold)[selected_features]
        X_unl_scaled = scaler.transform(X_unl_sample)
        unlabeled_fold_preds[:, i] = bootstrap_model.predict_proba(X_unl_scaled)[:, 1]
    
    # Calculate mean predictions
    mean_pos_preds = np.mean(fold_predictions, axis=1)
    mean_unl_preds = np.mean(unlabeled_fold_preds, axis=1)
    
    # Calculate fold accuracy (percentage of positive samples correctly classified as positive)
    fold_accuracy = (mean_pos_preds > threshold).mean()
    cv_accuracies.append(fold_accuracy)
    
    # Calculate AUC for this fold
    fold_all_probs = np.concatenate([mean_pos_preds, mean_unl_preds])
    fold_all_labels = np.concatenate([np.ones(len(mean_pos_preds)), np.zeros(len(mean_unl_preds))])
    fold_auc = roc_auc_score(fold_all_labels, fold_all_probs)
    cv_aucs.append(fold_auc)
    
    # Calculate precision and recall
    y_pred = (fold_all_probs > threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score
    fold_precision = precision_score(fold_all_labels, y_pred)
    fold_recall = recall_score(fold_all_labels, y_pred)
    cv_precision.append(fold_precision)
    cv_recall.append(fold_recall)
    
    print(f"    Fold {fold+1} - Accuracy: {fold_accuracy:.4f}, AUC: {fold_auc:.4f}")
    print(f"    Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}")

print(f"\nCross-validation results:")
print(f"Mean accuracy: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})")
print(f"Mean AUC: {np.mean(cv_aucs):.4f} (±{np.std(cv_aucs):.4f})")
print(f"Mean precision: {np.mean(cv_precision):.4f} (±{np.std(cv_precision):.4f})")
print(f"Mean recall: {np.mean(cv_recall):.4f} (±{np.std(cv_recall):.4f})")
# =======================================
# FEATURE IMPORTANCE ANALYSIS
# =======================================

print("\nAnalyzing feature importance...")

# Sort features by importance
feature_importance_df = pd.DataFrame({
    'feature': selected_features,
    'importance': feature_importances
})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

print("Top 15 most important features:")
for i in range(min(15, len(feature_importance_df))):
    print(f"{i+1}. {feature_importance_df.iloc[i]['feature']}: {feature_importance_df.iloc[i]['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15))
plt.title('Feature Importance in PU Learning Model')
plt.tight_layout()
plt.savefig('pu_feature_importance.png')
print("Feature importance plot saved as 'pu_feature_importance.png'")

# =======================================
# IDENTIFY TOP STRONG STOCKS
# =======================================

print("\nIdentifying top intrinsically strong stocks...")

# Extract the top strong stocks with high confidence
top_strong_stocks = final_predictions[
    (final_predictions['is_strong'] == 1)
].sort_values('confidence_score', ascending=False)

print(f"\nFound {len(top_strong_stocks)} stocks classified as 'Strong'")
print(f"With high confidence: {top_strong_stocks['high_confidence'].sum()} stocks")

# Print top 20 stocks
print("\n===== TOP 20 INTRINSICALLY STRONG STOCKS (WITH CONFIDENCE SCORES) =====")
top_20_strong = top_strong_stocks.head(20)
for i, (idx, row) in enumerate(top_20_strong.iterrows()):
    print(f"{i+1}. {row['symbol']} - {row['company_name']}")
    print(f"   Confidence Score: {row['confidence_score']:.4f} (±{row['uncertainty']:.4f})")
    print(f"   Market Cap Category: {row['market_cap_category']}")
    print(f"   Industry: {row['company_industry']}")
    print(f"   Sector: {row['sector']}")
    print(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}")
    print(f"   Return on Equity: {row['return_on_equity']:.4f}")
    print(f"   Profit Margin: {row['profit_margin']:.4f}")

# =======================================
# SECTOR ANALYSIS OF STRONG STOCKS
# =======================================

print("\nPerforming sector analysis of identified strong stocks...")

# Count of strong stocks by sector
sector_counts = top_strong_stocks['sector'].value_counts()
top_sectors = sector_counts.head(5)

print("\nTop 5 sectors with most strong stocks:")
for sector, count in top_sectors.items():
    percent = count / len(top_strong_stocks) * 100
    print(f"{sector}: {count} stocks ({percent:.1f}%)")

# =======================================
# CLASSIFICATION STABILITY ANALYSIS
# =======================================

print("\nAnalyzing classification stability...")

# Calculate how many times each stock flips between strong/not strong
stability_scores = np.zeros(len(unlabeled_df))
for i in range(1, n_bootstraps):
    pred_change = (bootstrap_predictions[:, i] > threshold).astype(int) != \
                  (bootstrap_predictions[:, i-1] > threshold).astype(int)
    stability_scores += pred_change

unlabeled_df['classification_flips'] = stability_scores
avg_flips = stability_scores.mean()
print(f"Average classification flips per stock: {avg_flips:.2f} out of {n_bootstraps-1} comparisons")
print(f"Stocks with perfect stability (0 flips): {(stability_scores == 0).sum()}")
print(f"Stocks with high instability (>25% flips): {(stability_scores > (n_bootstraps-1)/4).sum()}")

# Add high stability flag
unlabeled_df['stable_classification'] = (stability_scores < (n_bootstraps-1)/10)
print(f"Stocks with stable classification: {unlabeled_df['stable_classification'].sum()}")

# Update predictions with stability information
final_predictions['stable_classification'] = unlabeled_df['stable_classification']
final_predictions['classification_flips'] = unlabeled_df['classification_flips']

# =======================================
# VALIDATION ON HELD-OUT POSITIVES
# =======================================

print("\nValidating on held-out positive samples...")

# Scale the validation features
X_validation = positive_validation_df[selected_features]
X_validation_scaled = scaler.transform(X_validation)

# Initialize prediction array
validation_predictions = np.zeros((len(positive_validation_df), n_bootstraps))

# Get predictions from each bootstrap model
for i in range(n_bootstraps):
    bootstrap_model = train_bootstrap_model(positive_train_df, unlabeled_df, i)
    validation_predictions[:, i] = bootstrap_model.predict_proba(X_validation_scaled)[:, 1]

# Calculate mean predictions
positive_validation_df['mean_probability'] = np.mean(validation_predictions, axis=1)
positive_validation_df['std_probability'] = np.std(validation_predictions, axis=1)

# Calculate accuracy at the threshold used for classification
validation_accuracy = (positive_validation_df['mean_probability'] > threshold).mean()
print(f"Validation accuracy on held-out positive samples: {validation_accuracy:.4f}")
print(f"Percentage of known strong stocks correctly identified: {validation_accuracy*100:.1f}%")

# Combine validation data for ROC curve
validation_unlabeled = unlabeled_df.sample(len(positive_validation_df), random_state=42)
all_validation_probs = np.concatenate([
    positive_validation_df['mean_probability'].values,
    validation_unlabeled['mean_probability'].values
])
all_validation_labels = np.concatenate([
    np.ones(len(positive_validation_df)),
    np.zeros(len(validation_unlabeled))
])

# Calculate AUC and other metrics
validation_auc = roc_auc_score(all_validation_labels, all_validation_probs)
print(f"Validation AUC: {validation_auc:.4f}")

# Calculate optimal threshold based on validation data
precision, recall, thresholds = precision_recall_curve(all_validation_labels, all_validation_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold based on validation: {optimal_threshold:.4f} (current: {threshold:.4f})")

# =======================================
# VISUALIZATION OF PROBABILITY DISTRIBUTIONS
# =======================================

plt.figure(figsize=(12, 6))

# Plot distribution for unlabeled data
sns.histplot(unlabeled_df['mean_probability'], bins=30, alpha=0.6, label='Unlabeled Data')

# Plot distribution for held-out positive samples
sns.histplot(positive_validation_df['mean_probability'], bins=30, alpha=0.6, label='Known Strong Stocks', color='green')

# Add the threshold line
plt.axvline(x=threshold, color='red', linestyle='--', label=f'Current Threshold ({threshold:.2f})')
plt.axvline(x=optimal_threshold, color='orange', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.title('Distribution of Prediction Scores for Validation')
plt.xlabel('Mean Prediction Probability')
plt.ylabel('Count')
plt.legend()
plt.savefig('validation_probability_distribution.png')
print("Validation probability distribution saved as 'validation_probability_distribution.png'")

# ROC curve plot
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(all_validation_labels, all_validation_probs)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {validation_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Stock Classification Validation')
plt.legend()
plt.savefig('validation_roc_curve.png')
print("ROC curve saved as 'validation_roc_curve.png'")

# Plot sector distribution
plt.figure(figsize=(12, 6))
sector_counts.head(10).plot(kind='bar')
plt.title('Distribution of Strong Stocks by Sector')
plt.ylabel('Number of Stocks')
plt.tight_layout()
plt.savefig('strong_stocks_by_sector.png')
print("Sector analysis plot saved as 'strong_stocks_by_sector.png'")

# =======================================
# SAVE RESULTS
# =======================================

print("\nSaving results to CSV files...")

# Save all predictions
final_predictions[['symbol', 'company_name', 'company_industry', 'sector', 
                   'market_cap_category', 'predicted_label', 'confidence_score', 
                   'uncertainty', 'high_confidence', 'stable_classification']].to_csv('all_stock_predictions.csv', index=False)
print("All predictions saved to 'all_stock_predictions.csv'")

# Save top strong stocks - Fix for missing columns
# First check which columns actually exist in top_strong_stocks
columns_to_save = ['symbol', 'company_name', 'company_industry', 'sector',
                   'market_cap_category', 'confidence_score', 'uncertainty',
                   'high_confidence', 'sharpe_ratio', 'return_on_equity',
                   'profit_margin']

# Add optional columns if they exist
if 'stable_classification' in top_strong_stocks.columns:
    columns_to_save.append('stable_classification')
if 'classification_flips' in top_strong_stocks.columns:
    columns_to_save.append('classification_flips')
if 'year_change_percent' in top_strong_stocks.columns:
    columns_to_save.append('year_change_percent')

# Save only columns that exist
top_strong_stocks[columns_to_save].head(100).to_csv('top_100_strong_stocks.csv', index=False)
print("Top 100 strong stocks saved to 'top_100_strong_stocks.csv'")

# =======================================
# GENERATE COMPREHENSIVE REPORT
# =======================================

print("\nGenerating comprehensive analysis report...")

with open('pu_learning_stock_analysis_report.txt', 'w') as f:
    f.write("POSITIVE-UNLABELED LEARNING STOCK CLASSIFICATION REPORT\n")
    f.write("===================================================\n\n")
    
    f.write("ANALYSIS SUMMARY\n")
    f.write("--------------\n")
    f.write(f"Total companies analyzed: {len(df)}\n")
    f.write(f"Known strong companies (IN-L, IN-M): {len(positive_df)}\n")
    f.write(f"Unlabeled companies analyzed: {len(unlabeled_df)}\n")
    f.write(f"Companies classified as intrinsically strong: {len(top_strong_stocks)} ({len(top_strong_stocks)/len(unlabeled_df)*100:.1f}% of unlabeled)\n")
    f.write(f"High confidence strong predictions: {top_strong_stocks['high_confidence'].sum()} ({top_strong_stocks['high_confidence'].sum()/len(top_strong_stocks)*100:.1f}% of strong)\n\n")
    f.write("VALIDATION RESULTS\n")
    f.write("----------------\n")
    f.write(f"Cross-validation (5-fold) metrics:\n")
    f.write(f"  - Mean accuracy: {np.mean(cv_accuracies):.4f} (±{np.std(cv_accuracies):.4f})\n")
    f.write(f"  - Mean AUC: {np.mean(cv_aucs):.4f} (±{np.std(cv_aucs):.4f})\n")
    f.write(f"  - Mean precision: {np.mean(cv_precision):.4f} (±{np.std(cv_precision):.4f})\n")
    f.write(f"  - Mean recall: {np.mean(cv_recall):.4f} (±{np.std(cv_recall):.4f})\n\n")

    f.write(f"Hold-out validation metrics:\n")
    f.write(f"  - Accuracy on held-out positive samples: {validation_accuracy:.4f}\n")
    f.write(f"  - AUC: {validation_auc:.4f}\n")
    f.write(f"  - Current threshold: {threshold:.4f}\n")
    f.write(f"  - Optimal threshold from validation: {optimal_threshold:.4f}\n\n")

    f.write(f"Classification stability metrics:\n")
    f.write(f"  - Average classification flips: {avg_flips:.2f} out of {n_bootstraps-1}\n")
    f.write(f"  - Stocks with perfect stability: {(stability_scores == 0).sum()}\n")
    f.write(f"  - Stocks with high stability (< 10% flips): {unlabeled_df['stable_classification'].sum()}\n")
    f.write(f"  - Stocks with high instability (> 25% flips): {(stability_scores > (n_bootstraps-1)/4).sum()}\n\n")

    f.write(f"Top recommendations have high validation confidence:\n")
    # Safely check for highly confident stocks
    confidence_condition = top_strong_stocks['confidence_score'] > 0.8
    uncertainty_condition = top_strong_stocks['uncertainty'] < 0.1
    
    if 'stable_classification' in top_strong_stocks.columns:
        stable_condition = top_strong_stocks['stable_classification']
        highly_confident = top_strong_stocks[confidence_condition & uncertainty_condition & stable_condition].shape[0]
        f.write(f"  - {highly_confident} stocks have high confidence score, low uncertainty, and stable classification\n")
    else:
        highly_confident = top_strong_stocks[confidence_condition & uncertainty_condition].shape[0]
        f.write(f"  - {highly_confident} stocks have high confidence score and low uncertainty\n")
    
    f.write(f"  - These stocks represent the highest quality recommendations\n\n")
    f.write("MODEL INFORMATION\n")
    f.write("---------------\n")
    f.write(f"Learning approach: Positive-Unlabeled Learning with Bagging\n")
    f.write(f"Bootstrap iterations: {n_bootstraps}\n")
    f.write(f"Classification threshold: {threshold}\n")
    f.write(f"High confidence threshold: 0.75 (probability) and 0.15 (uncertainty)\n\n")
    
    f.write("TOP FEATURE IMPORTANCE\n")
    f.write("-------------------\n")
    for i in range(min(15, len(feature_importance_df))):
        f.write(f"{i+1}. {feature_importance_df.iloc[i]['feature']}: {feature_importance_df.iloc[i]['importance']:.4f}\n")
    f.write("\n")
    
    f.write("SECTOR ANALYSIS\n")
    f.write("-------------\n")
    f.write("Distribution of strong stocks by sector:\n")
    for sector, count in sector_counts.items():
        percent = count / len(top_strong_stocks) * 100
        f.write(f"{sector}: {count} stocks ({percent:.1f}%)\n")
    f.write("\n")
    3
    f.write("TOP 50 INTRINSICALLY STRONG STOCKS\n")
    f.write("-----------------------------\n")
    for i, (idx, row) in enumerate(top_strong_stocks.head(50).iterrows()):
        f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
        f.write(f"   Confidence Score: {row['confidence_score']:.4f} (±{row['uncertainty']:.4f})\n")
        f.write(f"   High Confidence: {'Yes' if row['high_confidence'] else 'No'}\n")
        f.write(f"   Market Cap: {row['market_cap_category']}\n")
        f.write(f"   Industry: {row['company_industry']}\n")
        f.write(f"   Sector: {row['sector']}\n")
        f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
        f.write(f"   Return on Equity: {row['return_on_equity']:.4f}\n")
        f.write(f"   Profit Margin: {row['profit_margin']:.4f}\n")
        # Only include year_change_percent if it exists
        if 'year_change_percent' in row:
            f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n\n")
        else:
            f.write("\n")

print("\nAnalysis complete! Results available in:")
print("1. all_stock_predictions.csv - All stock predictions with confidence scores")
print("2. top_100_strong_stocks.csv - Top 100 strong stocks with detailed metrics")
print("3. pu_learning_stock_analysis_report.txt - Comprehensive analysis report")
print("4. pu_feature_importance.png - Feature importance visualization")
print("5. strong_stocks_by_sector.png - Sector distribution of strong stocks")