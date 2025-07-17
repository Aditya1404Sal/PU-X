import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('nse_companies.csv')

# Explore the dataset
print(f"Total companies: {len(df)}")
print(f"Labeled companies: {sum(df['intrinsic_label'].notna() & (df['intrinsic_label'] != 'Unknown'))}")
print(f"Unlabeled companies: {sum(df['intrinsic_label'].isna() | (df['intrinsic_label'] == 'Unknown'))}")

# Analyze the distribution of labeled data
labeled_df = df[df['intrinsic_label'].notna() & (df['intrinsic_label'] != 'Unknown')]
print("\nDistribution of labeled data:")
print(labeled_df['intrinsic_label'].value_counts())

# Replace 'Unknown' with NaN for clarity
df['intrinsic_label'] = df['intrinsic_label'].replace('Unknown', np.nan)

# Feature Engineering
print("\nPerforming feature engineering...")
# Create financial ratios (with error handling for division by zero)
df['profit_margin'] = df['avg_profit_loss_for_period'] / df['avg_revenue_from_operations'].replace(0, np.nan)
df['return_on_equity'] = df['avg_profit_loss_for_period'] / df['avg_paid_up_equity_share_capital'].replace(0, np.nan)
df['debt_to_income'] = df['avg_finance_costs'] / df['avg_income'].replace(0, np.nan)
df['operational_efficiency'] = (df['avg_employee_benefit_expense'] + df['avg_other_expenses']) / df['avg_revenue_from_operations'].replace(0, np.nan)
df['earnings_growth'] = df['avg_comprehensive_income'] / df['avg_paid_up_equity_share_capital'].replace(0, np.nan)

# Create market performance metrics
df['price_stability'] = df['stability_score'] / df['volatility'].replace(0, np.nan)
df['risk_adjusted_return'] = df['sharpe_ratio'] * df['year_change_percent']

# Convert market_cap_category to numeric (for calculations)
market_cap_mapping = {'Large': 4, 'Mid': 3, 'Small': 2, 'Micro': 1}
df['market_cap_numeric'] = df['market_cap_category'].map(market_cap_mapping).fillna(0)

df['volume_to_market_cap'] = df['avg_volume'] / df['market_cap_numeric'].replace(0, np.nan)
df['price_to_earnings'] = df['avg_close'] / df['avg_basic_earnings_per_share'].replace(0, np.nan)

# Define financial health metrics
print("Calculating financial health metrics...")
# Profitability
df['profitability'] = df['avg_profit_loss_for_period'] / df['avg_revenue_from_operations'].replace(0, np.nan)

# Growth
df['revenue_to_market_cap'] = df['avg_revenue_from_operations'] / df['market_cap_numeric'].replace(0, np.nan)

# Stability
df['earnings_stability'] = df['stability_score'] * df['sharpe_ratio']

# Return on investment
df['roi_potential'] = df['avg_basic_earnings_per_share'] / df['avg_close'].replace(0, np.nan)

# Combine into health score
df['financial_health'] = (
    df['profitability'].fillna(0) * 0.3 +
    df['revenue_to_market_cap'].fillna(0) * 0.2 +
    df['earnings_stability'].fillna(0) * 0.3 +
    df['roi_potential'].fillna(0) * 0.2
)

# Financial score for performance
df['financial_score'] = (
    df['profit_margin'].fillna(0) + 
    df['return_on_equity'].fillna(0) + 
    df['sharpe_ratio'].fillna(0) -
    df['volatility'].fillna(0) / 100
)

# Feature selection for the model
features = [
    'avg_revenue_from_operations', 'avg_profit_loss_for_period', 'avg_basic_earnings_per_share',
    'avg_comprehensive_income', 'avg_finance_costs', 'profit_margin', 'return_on_equity', 
    'debt_to_income', 'operational_efficiency', 'earnings_growth', 'volatility', 
    'sharpe_ratio', 'year_change_percent', 'stability_score', 'trend_strength', 
    'liquidity_score', 'price_stability', 'risk_adjusted_return', 'financial_health',
    'financial_score'
]

# Handle missing values
print("Handling missing values...")
for feature in features:
    if feature in df.columns:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)

# Split into labeled and unlabeled datasets
print("Splitting data into labeled and unlabeled sets...")
# Only include rows where intrinsic_label is one of the valid labels we're looking for
valid_labels = ['IN-L', 'IN-M', 'IN-S', 'IN-Mi']
labeled_df = df[df['intrinsic_label'].isin(valid_labels)]
X_labeled = labeled_df[features]
y_labeled = labeled_df['intrinsic_label']

# Get unlabeled data
unlabeled_df = df[~df['intrinsic_label'].isin(valid_labels)]
X_unlabeled = unlabeled_df[features]
unlabeled_indices = unlabeled_df.index

# Convert categorical labels to numerical
print("Converting labels to numerical format...")
label_mapping = {'IN-L': 0, 'IN-M': 1, 'IN-S': 2, 'IN-Mi': 3}
reverse_mapping = {0: 'NIN-L', 1: 'NIN-M', 2: 'NIN-S', 3: 'NIN-Micro'}

# Convert labels and handle any unexpected values
y_labeled_num = pd.Series(index=y_labeled.index)
for i, label in enumerate(y_labeled):
    y_labeled_num.iloc[i] = label_mapping.get(label, np.nan)

# Remove any rows with NaN values in the target
valid_indices = ~y_labeled_num.isna()
X_labeled = X_labeled[valid_indices]
y_labeled_num = y_labeled_num[valid_indices].astype(int)

print(f"Training with {len(X_labeled)} valid labeled samples")

# Normalize the features
print("Normalizing features...")
scaler = StandardScaler()
X_labeled_scaled = scaler.fit_transform(X_labeled)
X_unlabeled_scaled = scaler.transform(X_unlabeled)

# Train a random forest classifier on the labeled data
print("\nTraining Random Forest classifier...")
X_train, X_test, y_train, y_test = train_test_split(X_labeled_scaled, y_labeled_num, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
print("\nEvaluating model performance:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = clf.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
features_df = features_df.sort_values('Importance', ascending=False)
print("\nTop 10 most important features:")
print(features_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
features_df.sort_values('Importance').plot(kind='barh', x='Feature', y='Importance')
plt.title('Feature Importance in Random Forest Model')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# Predict probabilities for unlabeled data
print("\nPredicting probabilities for unlabeled data...")
unlabeled_probas = clf.predict_proba(X_unlabeled_scaled)

# Get top candidates for each category
print("Finding top candidates for each category...")
top_candidates = {}

for class_idx in range(4):
    # Get probabilities for this class
    class_probas = unlabeled_probas[:, class_idx]
    
    # Find candidates with highest probability for this class
    # (We'll take more than 50 initially, then filter by financial performance)
    top_count = min(100, len(class_probas))
    top_indices = np.argsort(class_probas)[-top_count:]
    
    # Store original indices and probabilities
    top_candidates[reverse_mapping[class_idx]] = {
        'indices': [unlabeled_indices[i] for i in top_indices],
        'probabilities': class_probas[top_indices]
    }

# Filter top candidates by financial performance
print("Filtering candidates by financial performance...")
final_selections = {}

for label, candidates in top_candidates.items():
    indices = candidates['indices']
    probas = candidates['probabilities']
    
    # Get candidates with their financial scores
    candidates_df = pd.DataFrame({
        'index': indices,
        'probability': probas,
        'financial_score': df.loc[indices, 'financial_score'].values,
        'financial_health': df.loc[indices, 'financial_health'].values,
        'symbol': df.loc[indices, 'symbol'].values,
        'company_name': df.loc[indices, 'company_name'].values
    })
    
    # Sort by both classification probability and financial metrics
    candidates_df['combined_score'] = (
        candidates_df['probability'] * 0.5 + 
        (candidates_df['financial_score'] - candidates_df['financial_score'].min()) / 
        (candidates_df['financial_score'].max() - candidates_df['financial_score'].min() + 1e-10) * 0.3 +
        (candidates_df['financial_health'] - candidates_df['financial_health'].min()) / 
        (candidates_df['financial_health'].max() - candidates_df['financial_health'].min() + 1e-10) * 0.2
    )
    
    candidates_df = candidates_df.sort_values('combined_score', ascending=False)
    # Select top 50 for each category
    final_selections[label] = candidates_df.head(50)

# Update the original dataframe with new labels
print("\nAssigning new labels to selected companies...")
for label, selections in final_selections.items():
    symbols = selections['symbol'].tolist()
    for symbol in symbols:
        idx = df[df['symbol'] == symbol].index
        if len(idx) > 0:
            df.loc[idx, 'intrinsic_label'] = label

# Save the updated dataset
print("Saving updated dataset to 'nse_companies_labeled.csv'...")
df.to_csv('nse_companies_labeled.csv', index=False)

# Print summary of the selections
print("\n===== FINAL SELECTIONS =====")
for label, selections in final_selections.items():
    print(f"\nTop 10 {label} companies:")
    for i in range(min(10, len(selections))):
        row = selections.iloc[i]
        print(f"{row['symbol']} - {row['company_name']} (Score: {row['combined_score']:.4f})")

# Create a detailed report
print("\nCreating detailed report...")
with open('stock_selection_report.txt', 'w') as f:
    f.write("NSE STOCK SELECTION REPORT\n")
    f.write("=======================\n\n")
    
    f.write("MODEL PERFORMANCE\n")
    f.write("-----------------\n")
    f.write(classification_report(y_test, y_pred))
    f.write("\n\n")
    
    f.write("TOP FEATURES BY IMPORTANCE\n")
    f.write("-------------------------\n")
    for idx, row in features_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n\n")
    
    f.write("SELECTED COMPANIES BY CATEGORY\n")
    f.write("----------------------------\n")
    for label, selections in final_selections.items():
        f.write(f"\n{label} - TOP 50 COMPANIES:\n")
        for i in range(len(selections)):
            row = selections.iloc[i]
            f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
            f.write(f"   Combined Score: {row['combined_score']:.4f}\n")
            f.write(f"   Classification Probability: {row['probability']:.4f}\n")
            f.write(f"   Financial Score: {row['financial_score']:.4f}\n")
            f.write(f"   Financial Health: {row['financial_health']:.4f}\n")
        f.write("\n")

print("Detailed report saved as 'stock_selection_report.txt'")
print("\nAnalysis complete! You now have:")
print("1. Updated CSV with new labels for high-performing companies")
print("2. A feature importance plot showing the most predictive features")
print("3. A detailed report of all selected companies with their scores")