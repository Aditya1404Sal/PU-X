import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import time
from scipy.stats import entropy
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

# 6. Transform market_cap_category to numeric - using full integer values
market_cap_mapping = {'Large': 4, 'Mid': 3, 'Small': 2, 'Micro': 1, np.nan: 0}
df['market_cap_numeric'] = df['market_cap_category'].map(market_cap_mapping)

# 7. Size metrics with log transformations to handle skewness
df['log_revenue'] = np.log1p(df['avg_revenue_from_operations'])
df['log_profit'] = np.log1p(df['avg_profit_loss_for_period'].clip(lower=0))
df['log_market_cap'] = np.log1p(df['issued_size'] * df['avg_close'])  # More direct measure of market cap

# 8. Combined metrics
df['financial_health'] = (
    df['profit_margin'].fillna(0) + 
    df['return_on_equity'].fillna(0) + 
    df['sharpe_ratio'].fillna(0) -
    df['volatility'].fillna(0) / 100
)

# 9. Piotroski F-Score Components 
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

# Define candidate features - EXCLUDE market_cap_numeric to avoid bias
features = [
    # Profitability
    'profit_margin', 'gross_profit_margin', 'return_on_equity', 'return_on_assets', 'operating_margin',
    
    # Efficiency
    'asset_turnover', 'operational_efficiency', 'employee_productivity',
    
    # Leverage
    'debt_to_income', 'interest_coverage',
    
    # Valuation
    'earnings_growth', 'price_to_earnings', 'price_to_book',
    
    # Market metrics (excluding direct market cap)
    'volume_price_ratio', 'high_low_ratio', 'price_volatility',
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
# MULTICLASS PU LEARNING IMPLEMENTATION
# =======================================

print("\nImplementing Multiclass Positive-Unlabeled Learning for stock classification...")

# Step 1: Extract strength categories from intrinsic labels
# Parse intrinsic labels to get correct categories
df['strength_category'] = df['intrinsic_label'].apply(
    lambda x: x.split('-')[1] if isinstance(x, str) and x.startswith('IN-') else 'Unknown'
)
print("\nDistribution of strength categories:")
print(df['strength_category'].value_counts(dropna=False))

# Create numeric encoding for strength categories
category_map = {'L': 4, 'M': 3, 'S': 2, 'Mi': 1, 'Unknown': 0}
df['strength_category_numeric'] = df['strength_category'].map(category_map)

# Step 2: Separate labeled and unlabeled data
labeled_df = df[df['strength_category'] != 'Unknown'].copy()
unlabeled_df = df[df['strength_category'] == 'Unknown'].copy()

# Create class-specific dataframes
class_L_df = labeled_df[labeled_df['strength_category'] == 'L'].copy()
class_M_df = labeled_df[labeled_df['strength_category'] == 'M'].copy()
class_S_df = labeled_df[labeled_df['strength_category'] == 'S'].copy()
class_Mi_df = labeled_df[labeled_df['strength_category'] == 'Mi'].copy()

print(f"Labeled samples distribution:")
print(f"- Class L (Large): {len(class_L_df)} samples")
print(f"- Class M (Mid): {len(class_M_df)} samples")
print(f"- Class S (Small): {len(class_S_df)} samples")
print(f"- Class Mi (Micro): {len(class_Mi_df)} samples")
print(f"Total labeled: {len(labeled_df)} samples")
print(f"Unlabeled samples: {len(unlabeled_df)} samples")

# Step 3: Scale features
print("\nScaling features...")
scaler = RobustScaler()
X_labeled = labeled_df[features]
X_labeled_scaled = scaler.fit_transform(X_labeled)
X_unlabeled = unlabeled_df[features]
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Step 4: Implement separate PU learning for each market cap category
print("\nImplementing separate PU learning models for each market cap category...")

# Define a function to train a binary PU classifier for each class
def train_pu_classifier(positive_df, unlabeled_df, market_cap_category, features):
    """
    Train a PU classifier for a specific market cap category
    """
    print(f"\nTraining PU classifier for {market_cap_category} category...")
    
    # Filter unlabeled data by market cap to match the target category
    target_unlabeled_df = unlabeled_df[unlabeled_df['market_cap_category'] == market_cap_category].copy()
    
    if len(target_unlabeled_df) == 0:
        print(f"No unlabeled samples for {market_cap_category} category, skipping...")
        return None, None, None, None
    
    print(f"- Positive samples: {len(positive_df)}")
    print(f"- Target unlabeled samples: {len(target_unlabeled_df)}")
    
    # Scale features for this specific training set
    X_positive = positive_df[features]
    X_positive_scaled = scaler.transform(X_positive)
    
    X_target_unlabeled = target_unlabeled_df[features]
    X_target_unlabeled_scaled = scaler.transform(X_target_unlabeled)
    
    # Setup bagging parameters
    n_bootstraps = 100
    bootstrap_predictions = np.zeros((len(target_unlabeled_df), n_bootstraps))
    
    # Base classifier parameters - binary for this task
    base_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'learning_rate': 0.05,
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'min_child_weight': 3,
        'n_estimators': 100,
        'random_state': 42
    }
    
    # Storage for feature importance
    feature_importances = np.zeros(len(features))
    
    # Perform bagging iterations
    for i in range(n_bootstraps):
        if i % 10 == 0:
            print(f"  Bootstrap iteration {i+1}/{n_bootstraps}...")
        
        # Create balanced bootstrap sample
        positive_samples = min(len(positive_df), 50)  # Adjust as needed
        unlabeled_samples = positive_samples  # Equal number for balance
        
        # Sample from positive and unlabeled sets
        boot_positive = positive_df.sample(n=positive_samples, replace=True, random_state=i)
        boot_positive['label'] = 1  # Positive label
        
        boot_unlabeled = target_unlabeled_df.sample(n=min(unlabeled_samples, len(target_unlabeled_df)), 
                                                   replace=True, random_state=i)
        boot_unlabeled['label'] = 0  # Negative/unlabeled
        
        # Combine datasets
        bootstrap_df = pd.concat([boot_positive, boot_unlabeled])
        X_boot = bootstrap_df[features]
        y_boot = bootstrap_df['label']
        X_boot_scaled = scaler.transform(X_boot)
        
        # Train binary model
        bootstrap_model = xgb.XGBClassifier(**base_params)
        bootstrap_model.fit(X_boot_scaled, y_boot)
        
        # Store feature importances
        feature_importances += bootstrap_model.feature_importances_
        
        # Predict on all target unlabeled data
        bootstrap_predictions[:, i] = bootstrap_model.predict_proba(X_target_unlabeled_scaled)[:, 1]
    
    # Calculate average feature importance
    feature_importances /= n_bootstraps
    
    # Calculate prediction statistics
    target_unlabeled_df['mean_probability'] = np.mean(bootstrap_predictions, axis=1)
    target_unlabeled_df['std_probability'] = np.std(bootstrap_predictions, axis=1)
    target_unlabeled_df['confidence_score'] = target_unlabeled_df['mean_probability']
    target_unlabeled_df['uncertainty'] = target_unlabeled_df['std_probability']
    
    # Define high confidence predictions and is_strong flag
    target_unlabeled_df['high_confidence'] = (
        (target_unlabeled_df['confidence_score'] > 0.60) & 
        (target_unlabeled_df['uncertainty'] < 0.30)
    )
    target_unlabeled_df['is_strong'] = (target_unlabeled_df['mean_probability'] > 0.60).astype(int)
    
    # Return results and feature importances
    return target_unlabeled_df, bootstrap_predictions, feature_importances, features

# Train separate PU classifiers for each market cap category
results_L = train_pu_classifier(class_L_df, unlabeled_df, 'Large', features)
results_M = train_pu_classifier(class_M_df, unlabeled_df, 'Mid', features)
results_S = train_pu_classifier(class_S_df, unlabeled_df, 'Small', features)
results_Mi = train_pu_classifier(class_Mi_df, unlabeled_df, 'Micro', features)

# Collect predictions and feature importances
predictions_L, bootstrap_L, importances_L, _ = results_L if results_L[0] is not None else (None, None, None, None)
predictions_M, bootstrap_M, importances_M, _ = results_M if results_M[0] is not None else (None, None, None, None)
predictions_S, bootstrap_S, importances_S, _ = results_S if results_S[0] is not None else (None, None, None, None)
predictions_Mi, bootstrap_Mi, importances_Mi, _ = results_Mi if results_Mi[0] is not None else (None, None, None, None)

# Step 5: Combine and analyze results
print("\nCombining and analyzing results...")

# Initialize the combined prediction dataframe
all_predictions = pd.DataFrame()

# Add predictions from each category, if available
if predictions_L is not None:
    predictions_L['predicted_class'] = 'L'
    all_predictions = pd.concat([all_predictions, predictions_L])
    
if predictions_M is not None:
    predictions_M['predicted_class'] = 'M'
    all_predictions = pd.concat([all_predictions, predictions_M])
    
if predictions_S is not None:
    predictions_S['predicted_class'] = 'S'
    all_predictions = pd.concat([all_predictions, predictions_S])
    
if predictions_Mi is not None:
    predictions_Mi['predicted_class'] = 'Mi'
    all_predictions = pd.concat([all_predictions, predictions_Mi])

# Mark predictions as "strong" if confidence is high enough
if len(all_predictions) > 0:
    all_predictions['is_strong'] = (all_predictions['mean_probability'] > 0.60).astype(int)
    all_predictions['predicted_label'] = all_predictions.apply(
        lambda row: f"Strong-{row['predicted_class']}" if row['is_strong'] == 1 else "Not Strong",
        axis=1
    )

    print(f"\nTotal predictions generated: {len(all_predictions)}")
    print("Distribution of predicted labels:")
    print(all_predictions['predicted_label'].value_counts())

    # Filter for strong predictions only
    strong_predictions = all_predictions[all_predictions['is_strong'] == 1].copy()
    print(f"\nTotal strong predictions: {len(strong_predictions)}")
    print("Distribution of strong predictions by class:")
    print(strong_predictions['predicted_class'].value_counts())

# =======================================
# FEATURE IMPORTANCE ANALYSIS
# =======================================

print("\nAnalyzing feature importance by class...")

# Function to analyze and plot feature importance for a class
def analyze_feature_importance(importances, class_name):
    if importances is None:
        print(f"No feature importance data for class {class_name}")
        return None
    
    # Create dataframe of feature importances
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    print(f"\nTop 10 important features for {class_name}:")
    for i in range(min(10, len(importance_df))):
        print(f"{i+1}. {importance_df.iloc[i]['feature']}: {importance_df.iloc[i]['importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(15))
    plt.title(f'Feature Importance for Class {class_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{class_name}.png')
    print(f"Feature importance plot saved as 'feature_importance_{class_name}.png'")
    
    return importance_df

# Analyze feature importance for each class
importance_df_L = analyze_feature_importance(importances_L, 'L')
importance_df_M = analyze_feature_importance(importances_M, 'M')
importance_df_S = analyze_feature_importance(importances_S, 'S')
importance_df_Mi = analyze_feature_importance(importances_Mi, 'Mi')

# =======================================
# TOP STOCKS BY CLASS
# =======================================

print("\n===== TOP STOCKS BY STRENGTH CLASS (WITH CONFIDENCE SCORES) =====")

# Function to print top stocks for a class
def print_top_stocks(predictions_df, class_name, n=10):
    if predictions_df is None or len(predictions_df) == 0:
        print(f"\nNo predictions available for class {class_name}")
        return
    
    # Ensure 'is_strong' column exists
    if 'is_strong' not in predictions_df.columns:
        print(f"Warning: 'is_strong' column not found in {class_name} predictions, creating it")
        predictions_df['is_strong'] = (predictions_df['mean_probability'] > 0.60).astype(int)
    
    strong_df = predictions_df[predictions_df['is_strong'] == 1].sort_values('confidence_score', ascending=False)
    
    if len(strong_df) == 0:
        print(f"\nNo strong stocks identified for class {class_name}")
        return
    
    print(f"\n----- Top {min(n, len(strong_df))} Strong {class_name} Stocks -----")
    top_n = strong_df.head(n)
    for i, (idx, row) in enumerate(top_n.iterrows()):
        print(f"{i+1}. {row['symbol']} - {row['company_name']}")
        print(f"   Confidence Score: {row['confidence_score']:.4f} (±{row['uncertainty']:.4f})")
        print(f"   Market Cap Category: {row['market_cap_category']}")
        print(f"   Industry: {row['company_industry']}")
        print(f"   Sector: {row['sector']}")
        print(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}")
        print(f"   Return on Equity: {row['return_on_equity']:.4f}")
        print(f"   Profit Margin: {row['profit_margin']:.4f}")
        print(f"   Year Change %: {row['year_change_percent']:.2f}%")

# Print top stocks for each class
print_top_stocks(predictions_L, 'Large', n=10)
print_top_stocks(predictions_M, 'Mid', n=10)
print_top_stocks(predictions_S, 'Small', n=10)
print_top_stocks(predictions_Mi, 'Micro', n=10)

# =======================================
# SECTOR ANALYSIS BY CLASS
# =======================================

print("\nPerforming sector analysis by strength class...")

# Function to analyze sectors for a class
def analyze_sectors_by_class(predictions_df, class_name):
    if predictions_df is None or len(predictions_df) == 0:
        print(f"\nNo predictions available for class {class_name}")
        return
    
    # Ensure 'is_strong' column exists
    if 'is_strong' not in predictions_df.columns:
        print(f"Warning: 'is_strong' column not found in {class_name} predictions, creating it")
        predictions_df['is_strong'] = (predictions_df['mean_probability'] > 0.60).astype(int)
    
    strong_df = predictions_df[predictions_df['is_strong'] == 1]
    
    if len(strong_df) == 0:
        print(f"\nNo strong stocks identified for class {class_name}")
        return
    
    sector_counts = strong_df['sector'].value_counts()
    top_sectors = sector_counts.head(5)
    
    print(f"\nTop sectors for {class_name} strong stocks:")
    for sector, count in top_sectors.items():
        percent = count / len(strong_df) * 100
        print(f"{sector}: {count} stocks ({percent:.1f}%)")
    
    # Plot sector distribution
    plt.figure(figsize=(12, 6))
    sector_counts.head(10).plot(kind='bar')
    plt.title(f'Distribution of Strong {class_name} Stocks by Sector')
    plt.ylabel('Number of Stocks')
    plt.tight_layout()
    plt.savefig(f'strong_{class_name}_by_sector.png')
    print(f"Sector analysis plot saved as 'strong_{class_name}_by_sector.png'")

# Analyze sectors for each class
analyze_sectors_by_class(predictions_L, 'Large')
analyze_sectors_by_class(predictions_M, 'Mid')
analyze_sectors_by_class(predictions_S, 'Small')
analyze_sectors_by_class(predictions_Mi, 'Micro')

# =======================================
# SAVE RESULTS
# =======================================

print("\nSaving results to CSV files...")

# Save all predictions
if 'all_predictions' in locals() and len(all_predictions) > 0:
    all_predictions[['symbol', 'company_name', 'company_industry', 'sector', 
                    'market_cap_category', 'predicted_class', 'predicted_label',
                    'confidence_score', 'mean_probability', 'uncertainty', 
                    'high_confidence', 'is_strong']].to_csv('multiclass_pu_predictions.csv', index=False)
    print("All predictions saved to 'multiclass_pu_predictions.csv'")

# Save strong predictions by class
for class_name, pred_df in [('Large', predictions_L), ('Mid', predictions_M), 
                           ('Small', predictions_S), ('Micro', predictions_Mi)]:
    if pred_df is not None and len(pred_df) > 0:
        # Ensure 'is_strong' column exists
        if 'is_strong' not in pred_df.columns:
            pred_df['is_strong'] = (pred_df['mean_probability'] > 0.60).astype(int)
            
        strong_df = pred_df[pred_df['is_strong'] == 1].sort_values('confidence_score', ascending=False)
        if len(strong_df) > 0:
            strong_df[['symbol', 'company_name', 'company_industry', 'sector',
                      'market_cap_category', 'confidence_score', 'uncertainty',
                      'high_confidence', 'sharpe_ratio', 'return_on_equity',
                      'profit_margin', 'year_change_percent']].to_csv(f'strong_{class_name}_stocks.csv', index=False)
            print(f"Strong {class_name} stocks saved to 'strong_{class_name}_stocks.csv'")

# =======================================
# GENERATE COMPREHENSIVE REPORT
# =======================================


print("\nGenerating comprehensive analysis report...")

with open('multiclass_pu_stock_analysis_report.txt', 'w') as f:
    f.write("MULTICLASS POSITIVE-UNLABELED LEARNING STOCK CLASSIFICATION REPORT\n")
    f.write("==============================================================\n\n")
    
    f.write("ANALYSIS SUMMARY\n")
    f.write("--------------\n")
    f.write(f"Total companies analyzed: {len(df)}\n")
    f.write(f"Known labeled companies: {len(labeled_df)}\n")
    f.write(f"- Class L (Large): {len(class_L_df)}\n")
    f.write(f"- Class M (Mid): {len(class_M_df)}\n") 
    f.write(f"- Class S (Small): {len(class_S_df)}\n")
    f.write(f"- Class Mi (Micro): {len(class_Mi_df)}\n")
    f.write(f"Unlabeled companies analyzed: {len(unlabeled_df)}\n\n")
    
    f.write("MODEL INFORMATION\n")
    f.write("---------------\n")
    f.write("Learning approach: Class-specific Positive-Unlabeled Learning with Bagging\n")
    f.write("Method: Separate binary PU classifiers for each market cap category\n")
    f.write("Bootstrap iterations per class: 100\n")
    f.write("Strong classification threshold: 0.60 (probability)\n")
    f.write("High confidence threshold: 0.60 (confidence) and 0.30 (uncertainty)\n\n")
    
    f.write("CLASSIFICATION RESULTS\n")
    f.write("-------------------\n")
    if len(all_predictions) > 0:
        f.write("Distribution of predictions:\n")
        for label, count in all_predictions['predicted_label'].value_counts().items():
            percent = count / len(all_predictions) * 100
            f.write(f"- {label}: {count} stocks ({percent:.1f}%)\n")
    else:
        f.write("No predictions were generated.\n")
    f.write("\n")
    
    # Write feature importance for each class
    for class_name, importance_df in [('Large', importance_df_L), ('Mid', importance_df_M), 
                                    ('Small', importance_df_S), ('Micro', importance_df_Mi)]:
        if importance_df is not None:
            f.write(f"TOP FEATURE IMPORTANCE FOR CLASS {class_name}\n")
            f.write("-" * (len(f"TOP FEATURE IMPORTANCE FOR CLASS {class_name}") + 1) + "\n")
            for i in range(min(10, len(importance_df))):
                f.write(f"{i+1}. {importance_df.iloc[i]['feature']}: {importance_df.iloc[i]['importance']:.4f}\n")
            f.write("\n")
    
    # Write top stocks for each class
    for class_name, pred_df in [('Large', predictions_L), ('Mid', predictions_M), 
                               ('Small', predictions_S), ('Micro', predictions_Mi)]:
        if pred_df is not None and len(pred_df) > 0:
            strong_df = pred_df[pred_df['is_strong'] == 1].sort_values('confidence_score', ascending=False)
            if len(strong_df) > 0:
                f.write(f"TOP 20 STRONG {class_name.upper()} STOCKS\n")
                f.write("-" * (len(f"TOP 20 STRONG {class_name.upper()} STOCKS") + 1) + "\n")
                top_df = strong_df.head(20)
                for i, (idx, row) in enumerate(top_df.iterrows()):
                    f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
                    f.write(f"   Confidence Score: {row['confidence_score']:.4f} (±{row['uncertainty']:.4f})\n")
                    f.write(f"   High Confidence: {'Yes' if row['high_confidence'] else 'No'}\n")
                    f.write(f"   Market Cap: {row['market_cap_category']}\n")
                    f.write(f"   Industry: {row['company_industry']}\n")
                    f.write(f"   Sector: {row['sector']}\n")
                    f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
                    f.write(f"   Return on Equity: {row['return_on_equity']:.4f}\n")
                    f.write(f"   Profit Margin: {row['profit_margin']:.4f}\n")
                    f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n\n")

print("\nClass-specific PU Learning analysis complete! Results available in:")
print("1. multiclass_pu_predictions.csv - All stock predictions")
print("2. strong_*_stocks.csv files - Strong stocks identified for each category")
print("3. multiclass_pu_stock_analysis_report.txt - Comprehensive analysis report")
print("4. feature_importance_*.png - Feature importance visualizations by class")
print("5. strong_*_by_sector.png - Sector distribution for each category")