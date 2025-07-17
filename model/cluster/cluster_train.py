import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
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
    'profit_margin', 'return_on_equity', 'debt_to_income', 'operational_efficiency',
    'earnings_growth', 'volatility', 'sharpe_ratio', 'year_change_percent',
    'stability_score', 'trend_strength', 'liquidity_score', 'price_stability',
    'financial_health', 'financial_score'
]

# Handle missing values
print("Handling missing values...")
for feature in features:
    if feature in df.columns:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)

# Prepare data for clustering
print("Preparing data for clustering...")
X_all = df[features]
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Reduce dimensionality for visualization
print("Reducing dimensionality with PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_all_scaled)
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

# Create label mapping
label_mapping = {'IN-L': 0, 'IN-M': 1, 'IN-S': 2, 'IN-Mi': 3}
reverse_mapping = {0: 'NIN-L', 1: 'NIN-M', 2: 'NIN-S', 3: 'NIN-Micro'}

# Set up for clustering
print("\nPerforming KMeans clustering...")
n_clusters = 4  # We want 4 clusters: Large, Mid, Small, Micro
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_all_scaled)

# Calculate silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(X_all_scaled, df['cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Analyze clusters
print("\nCluster distribution:")
print(df['cluster'].value_counts())

# Map labeled data to clusters
labeled_df = df[df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])]
print("\nMapping labeled data to clusters:")
cluster_label_mapping = {}

# For each cluster, find the dominant label
for cluster_id in range(n_clusters):
    cluster_data = labeled_df[labeled_df['cluster'] == cluster_id]
    if len(cluster_data) > 0:
        # Count occurrences of each label in this cluster
        label_counts = cluster_data['intrinsic_label'].value_counts()
        dominant_label = label_counts.idxmax()
        cluster_label_mapping[cluster_id] = dominant_label
        print(f"Cluster {cluster_id} -> {dominant_label} (Contains {len(cluster_data)} labeled companies)")
        print(f"  Label distribution: {label_counts.to_dict()}")
    else:
        print(f"Cluster {cluster_id} has no labeled companies")

# If any cluster has no dominant label, assign based on financial metrics
for cluster_id in range(n_clusters):
    if cluster_id not in cluster_label_mapping:
        cluster_data = df[df['cluster'] == cluster_id]
        avg_market_cap = cluster_data['market_cap_numeric'].mean()
        
        if avg_market_cap >= 3.5:
            cluster_label_mapping[cluster_id] = 'NIN-L'
        elif avg_market_cap >= 2.5:
            cluster_label_mapping[cluster_id] = 'NIN-M'
        elif avg_market_cap >= 1.5:
            cluster_label_mapping[cluster_id] = 'NIN-S'
        else:
            cluster_label_mapping[cluster_id] = 'NIN-Micro'
            
        print(f"Assigned cluster {cluster_id} -> {cluster_label_mapping[cluster_id]} based on financial metrics")

# Visualize the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['pca_1'], df['pca_2'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')

# Mark labeled points
labeled_mask = df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])
plt.scatter(df[labeled_mask]['pca_1'], df[labeled_mask]['pca_2'], c='red', marker='x', alpha=0.5)

plt.title('PCA Visualization of Stock Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('stock_clusters.png')
print("Cluster visualization saved as 'stock_clusters.png'")

# Use clustering results to identify high-performing companies in each category

# Create a performance metric to rank companies within clusters
df['performance_score'] = (
    df['financial_score'] * 0.4 +
    df['financial_health'] * 0.3 +
    df['sharpe_ratio'].clip(-3, 3) * 0.2 +
    df['stability_score'] * 0.1
)

# Find unlabeled companies in each cluster
print("\nIdentifying top performing companies in each cluster...")
top_companies = {}

# For each cluster, find the top unlabeled companies by performance
for cluster_id, nin_label in cluster_label_mapping.items():
    # Get unlabeled companies in this cluster
    unlabeled_in_cluster = df[(df['cluster'] == cluster_id) & 
                             (~df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi']))]
    
    # Sort by performance score
    top_in_cluster = unlabeled_in_cluster.sort_values('performance_score', ascending=False)
    
    # Take top 50 or fewer if not enough companies
    top_50 = top_in_cluster.head(50)
    top_companies[nin_label] = top_50
    
    print(f"Selected {len(top_50)} companies for {nin_label}")

# Update the original dataframe with new labels
print("\nAssigning new labels to selected companies...")
for nin_label, companies in top_companies.items():
    for idx in companies.index:
        df.loc[idx, 'intrinsic_label'] = nin_label

# Save the updated dataset
print("Saving updated dataset to 'nse_companies_labeled.csv'...")
df.to_csv('nse_companies_labeled.csv', index=False)

# Print summary of the selections
print("\n===== FINAL SELECTIONS =====")
for nin_label, companies in top_companies.items():
    print(f"\nTop 10 {nin_label} companies:")
    for i in range(min(10, len(companies))):
        row = companies.iloc[i]
        print(f"{row['symbol']} - {row['company_name']} (Score: {row['performance_score']:.4f})")

# Create a detailed report
print("\nCreating detailed report...")
with open('stock_selection_report.txt', 'w') as f:
    f.write("NSE STOCK SELECTION REPORT (CLUSTERING-BASED)\n")
    f.write("=========================================\n\n")
    
    f.write("CLUSTERING RESULTS\n")
    f.write("-----------------\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n\n")
    
    for cluster_id, label in cluster_label_mapping.items():
        f.write(f"Cluster {cluster_id} -> {label}\n")
    f.write("\n\n")
    
    f.write("CLUSTER-LABEL DISTRIBUTION\n")
    f.write("-------------------------\n")
    for cluster_id in range(n_clusters):
        cluster_data = labeled_df[labeled_df['cluster'] == cluster_id]
        f.write(f"Cluster {cluster_id}:\n")
        if len(cluster_data) > 0:
            label_counts = cluster_data['intrinsic_label'].value_counts()
            for label, count in label_counts.items():
                f.write(f"  {label}: {count} companies\n")
        else:
            f.write("  No labeled companies\n")
        f.write("\n")
    
    f.write("SELECTED COMPANIES BY CATEGORY\n")
    f.write("----------------------------\n")
    for nin_label, companies in top_companies.items():
        f.write(f"\n{nin_label} - TOP 50 COMPANIES:\n")
        for i in range(len(companies)):
            row = companies.iloc[i]
            f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
            f.write(f"   Performance Score: {row['performance_score']:.4f}\n")
            f.write(f"   Financial Health: {row['financial_health']:.4f}\n")
            f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
            f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n")
        f.write("\n")

# Train a classifier to validate clustering results (optional)
print("\nTraining a classifier to validate clustering results...")
labeled_data = df[df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])]
X_labeled = labeled_data[features]
y_labeled = labeled_data['intrinsic_label'].map({'IN-L': 0, 'IN-M': 1, 'IN-S': 2, 'IN-Mi': 3})

X_train, X_test, y_train, y_test = train_test_split(
    scaler.transform(X_labeled), y_labeled, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassifier performance on labeled data:")
print(classification_report(y_test, y_pred))

# Calculate feature importances
feature_importance = clf.feature_importances_
features_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
features_df = features_df.sort_values('Importance', ascending=False)

print("\nTop 10 most important features:")
print(features_df.head(10))

# Add feature importance to report
with open('stock_selection_report.txt', 'a') as f:
    f.write("\nFEATURE IMPORTANCE\n")
    f.write("-----------------\n")
    for idx, row in features_df.head(10).iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

print("\nDetailed report saved as 'stock_selection_report.txt'")
print("\nAnalysis complete! You now have:")
print("1. Updated CSV with new labels for high-performing companies")
print("2. A cluster visualization showing how companies group together")
print("3. A detailed report of all selected companies with their scores")

# Generate an additional visualization showing performance distribution
plt.figure(figsize=(12, 8))
for i, label in enumerate(['IN-L', 'IN-M', 'IN-S', 'IN-Mi']):
    label_data = df[df['intrinsic_label'] == label]
    plt.scatter(label_data['financial_health'], label_data['performance_score'], 
                alpha=0.7, label=label)

for nin_label, companies in top_companies.items():
    plt.scatter(companies['financial_health'], companies['performance_score'], 
                alpha=0.7, marker='*', s=100, label=nin_label)

plt.title('Financial Health vs Performance Score')
plt.xlabel('Financial Health')
plt.ylabel('Performance Score')
plt.legend()
plt.savefig('performance_distribution.png')
print("Performance distribution visualization saved as 'performance_distribution.png'")