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

# Convert market_cap_category to numeric (for calculations)
market_cap_mapping = {'Large': 4, 'Mid': 3, 'Small': 2, 'Micro': 1, np.nan: 0}
df['market_cap_numeric'] = df['market_cap_category'].map(market_cap_mapping)

# Create financial health metrics
print("Calculating financial health metrics...")
# Profitability
df['profitability'] = df['avg_profit_loss_for_period'] / df['avg_revenue_from_operations'].replace(0, np.nan)

# Growth
df['revenue_to_market_cap'] = df['avg_revenue_from_operations'] / df['market_cap_numeric'].replace(0, np.nan)

# Stability
df['earnings_stability'] = df['stability_score'] * df['sharpe_ratio']

# Financial score for performance (will use later for ranking within clusters)
df['financial_score'] = (
    df['profit_margin'].fillna(0) + 
    df['return_on_equity'].fillna(0) + 
    df['sharpe_ratio'].fillna(0) -
    df['volatility'].fillna(0) / 100
)

# MODIFIED: Create more market cap related features to emphasize this characteristic
df['log_revenue'] = np.log1p(df['avg_revenue_from_operations'])
df['log_profit'] = np.log1p(df['avg_profit_loss_for_period'].clip(lower=0))
df['size_score'] = df['market_cap_numeric'] * 3 + df['log_revenue'] / 5

# MODIFIED: Feature selection with emphasis on market cap and company size
features = [
    'market_cap_numeric',   # Now given higher importance
    'size_score',           # New feature that emphasizes size
    'log_revenue',          # Log transformed for better scaling
    'log_profit',           # Log transformed for better scaling
    'avg_basic_earnings_per_share',
    'profit_margin',
    'return_on_equity',
    'stability_score',
    'sharpe_ratio'
]

# Handle missing values
print("Handling missing values...")
for feature in features:
    if feature in df.columns:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)

# MODIFIED: Feature weights (emphasize market cap features)
feature_weights = {
    'market_cap_numeric': 10.0,    # Increased from 5.0
    'size_score': 7.0,             # Increased from 4.0
    'log_revenue': 2.0,
    'log_profit': 1.5,
    'avg_basic_earnings_per_share': 0.5,
    'profit_margin': 0.3,          # Decreased importance
    'return_on_equity': 0.3,       # Decreased importance
    'stability_score': 0.3,        # Decreased importance
    'sharpe_ratio': 0.3            # Decreased importance
}

# Prepare data for clustering
print("Preparing data for clustering with weighted features...")
X_all = df[features]
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Apply feature weights
weighted_X = np.copy(X_all_scaled)
for i, feature in enumerate(features):
    weighted_X[:, i] *= feature_weights[feature]

# MODIFIED: Use more clusters to better capture the natural groupings
n_clusters = 4  # We want 4 clusters: Large, Mid, Small, Micro
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Fit KMeans on weighted features
print("\nPerforming KMeans clustering with weighted features...")
df['cluster'] = kmeans.fit_predict(weighted_X)

# Calculate silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(weighted_X, df['cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Analyze clusters
print("\nCluster distribution:")
print(df['cluster'].value_counts())

# Calculate average market cap numeric for each cluster
print("\nAverage market cap numeric by cluster:")
for cluster_id in range(n_clusters):
    avg_market_cap = df[df['cluster'] == cluster_id]['market_cap_numeric'].mean()
    print(f"Cluster {cluster_id}: {avg_market_cap:.2f}")

# MODIFIED: Map clusters to categories based on average market cap ranking
cluster_market_caps = []
for cluster_id in range(n_clusters):
    avg_market_cap = df[df['cluster'] == cluster_id]['market_cap_numeric'].mean()
    cluster_market_caps.append((cluster_id, avg_market_cap))

# Sort clusters by market cap (descending)
cluster_market_caps.sort(key=lambda x: x[1], reverse=True)

# Map clusters to labels based on sorted order
cluster_label_mapping = {
    cluster_market_caps[0][0]: 'NIN-L',   # Highest avg market cap -> Large
    cluster_market_caps[1][0]: 'NIN-M',   # Second highest -> Mid
    cluster_market_caps[2][0]: 'NIN-S',   # Third highest -> Small
    cluster_market_caps[3][0]: 'NIN-Mi'   # Lowest -> Micro
}

print("\nAutomatic cluster-to-label mapping based on market cap:")
for cluster_id, label in cluster_label_mapping.items():
    avg_market_cap = df[df['cluster'] == cluster_id]['market_cap_numeric'].mean()
    print(f"Cluster {cluster_id} -> {label} (Avg Market Cap: {avg_market_cap:.2f})")

# MODIFIED: Check label distribution in each cluster to verify
labeled_df = df[df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])]
print("\nLabel distribution in each cluster:")
for cluster_id in range(n_clusters):
    cluster_data = labeled_df[labeled_df['cluster'] == cluster_id]
    print(f"Cluster {cluster_id} ({cluster_label_mapping[cluster_id]}):")
    if len(cluster_data) > 0:
        label_counts = cluster_data['intrinsic_label'].value_counts()
        total = len(cluster_data)
        for label, count in label_counts.items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} companies ({percentage:.1f}%)")
    else:
        print("  No labeled companies")
    print()

# Reduce dimensionality for visualization
print("Reducing dimensionality with PCA for visualization...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(weighted_X)  # Use weighted features for PCA
df['pca_1'] = X_pca[:, 0]
df['pca_2'] = X_pca[:, 1]

# Visualize the clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['pca_1'], df['pca_2'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Cluster')

# Mark labeled points
labeled_mask = df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])
plt.scatter(df[labeled_mask]['pca_1'], df[labeled_mask]['pca_2'], c='red', marker='x', alpha=0.5)

# Add cluster labels to the plot
for cluster_id in range(n_clusters):
    cluster_points = df[df['cluster'] == cluster_id]
    centroid_x = cluster_points['pca_1'].mean()
    centroid_y = cluster_points['pca_2'].mean()
    plt.annotate(cluster_label_mapping[cluster_id], 
                 (centroid_x, centroid_y),
                 fontsize=14, 
                 fontweight='bold')

plt.title('PCA Visualization of Stock Clusters (Market Cap Weighted)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('stock_clusters.png')
print("Cluster visualization saved as 'stock_clusters.png'")

# Create a performance metric to rank companies within clusters
df['performance_score'] = (
    df['financial_score'] * 0.6 +
    df['sharpe_ratio'].clip(-3, 3) * 0.3 +
    df['stability_score'] * 0.1
)

# Find top performing companies in each cluster
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
    f.write("NSE STOCK SELECTION REPORT (MARKET CAP FOCUSED CLUSTERING)\n")
    f.write("=========================================\n\n")
    
    f.write("CLUSTERING RESULTS\n")
    f.write("-----------------\n")
    f.write(f"Silhouette Score: {silhouette_avg:.4f}\n\n")
    
    for cluster_id, label in cluster_label_mapping.items():
        avg_market_cap = df[df['cluster'] == cluster_id]['market_cap_numeric'].mean()
        f.write(f"Cluster {cluster_id} -> {label} (Avg Market Cap: {avg_market_cap:.2f})\n")
    f.write("\n\n")
    
    f.write("CLUSTER-LABEL DISTRIBUTION\n")
    f.write("-------------------------\n")
    for cluster_id in range(n_clusters):
        cluster_data = labeled_df[labeled_df['cluster'] == cluster_id]
        f.write(f"Cluster {cluster_id} ({cluster_label_mapping[cluster_id]}):\n")
        if len(cluster_data) > 0:
            label_counts = cluster_data['intrinsic_label'].value_counts()
            total = len(cluster_data)
            for label, count in label_counts.items():
                percentage = (count / total) * 100
                f.write(f"  {label}: {count} companies ({percentage:.1f}%)\n")
        else:
            f.write("  No labeled companies\n")
        f.write("\n")
    
    f.write("SELECTED COMPANIES BY CATEGORY\n")
    f.write("----------------------------\n")
    for nin_label, companies in top_companies.items():
        f.write(f"\n{nin_label} - TOP {len(companies)} COMPANIES:\n")
        for i in range(len(companies)):
            row = companies.iloc[i]
            f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
            f.write(f"   Performance Score: {row['performance_score']:.4f}\n")
            f.write(f"   Market Cap Category: {row['market_cap_category']}\n")
            f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
            f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n")
        f.write("\n")

# Train a classifier to validate clustering results
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

print("\nFeature importance ranking:")
for idx, row in features_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Add feature importance to report
with open('stock_selection_report.txt', 'a') as f:
    f.write("\nFEATURE IMPORTANCE\n")
    f.write("-----------------\n")
    for idx, row in features_df.iterrows():
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")

# Generate an additional visualization showing market cap distribution by cluster
plt.figure(figsize=(12, 8))
for cluster_id in range(n_clusters):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.hist(cluster_data['market_cap_numeric'], bins=20, alpha=0.5, 
             label=f"Cluster {cluster_id}: {cluster_label_mapping[cluster_id]}")

plt.title('Market Cap Distribution by Cluster')
plt.xlabel('Market Cap Numeric')
plt.ylabel('Count')
plt.legend()
plt.savefig('market_cap_distribution.png')
print("Market cap distribution visualization saved as 'market_cap_distribution.png'")

# Generate a visualization showing performance by market cap
plt.figure(figsize=(14, 8))
colors = ['b', 'g', 'r', 'c']
for i, (cluster_id, label) in enumerate(cluster_label_mapping.items()):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(cluster_data['market_cap_numeric'], cluster_data['performance_score'], 
                alpha=0.6, c=colors[i], label=label)

plt.title('Performance Score vs Market Cap')
plt.xlabel('Market Cap (Numeric)')
plt.ylabel('Performance Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('performance_by_market_cap.png')
print("Performance vs market cap visualization saved as 'performance_by_market_cap.png'")

print("\nAnalysis complete! You now have:")
print("1. Updated CSV with new labels for high-performing companies")
print("2. Multiple visualizations showing how companies cluster by market cap")
print("3. A detailed report of all selected companies with their scores")