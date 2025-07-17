import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, silhouette_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
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

# Feature Engineering - Enhanced with more financial ratios
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

# 6. Transform market_cap_category to numeric (for calculations)
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

# Create a list of all potential features
all_features = [
    # Size metrics
    'market_cap_numeric', 'log_revenue', 'log_profit', 'log_market_cap',
    
    # Profitability
    'profit_margin', 'gross_profit_margin', 'return_on_equity', 'return_on_assets', 'operating_margin',
    
    # Efficiency
    'asset_turnover', 'operational_efficiency', 'employee_productivity',
    
    # Leverage
    'debt_to_income', 'interest_coverage',
    
    # Valuation
    'earnings_growth', 'price_to_earnings', 'price_to_book',
    
    # Market performance
    'volume_price_ratio', 'high_low_ratio', 'price_volatility', 'volatility', 'sharpe_ratio',
    'avg_daily_return', 'avg_vwap_distance', 'avg_volume_spike', 'yearly_momentum_score',
    'liquidity_score', 'stability_score', 'trend_strength',
    
    # Basic financials
    'avg_basic_earnings_per_share', 'avg_diluted_earnings_per_share'
]

# Handle missing values with median imputation
print("Handling missing values...")
for feature in all_features:
    if feature in df.columns:
        median_value = df[feature].median()
        df[feature] = df[feature].fillna(median_value)

# Check for infinite values and replace with column max or min
for col in all_features:
    if col in df.columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        max_val = df[col].max()
        min_val = df[col].min()
        df[col] = df[col].fillna(max_val if max_val > 0 else min_val)

# Feature selection based on correlation to labels for labeled data
print("\nPerforming feature selection based on correlation to labels...")
# Create binary encoding for labeled data (for correlation analysis)
label_df = df[df['intrinsic_label'].notna()].copy()
label_df['is_large'] = (label_df['intrinsic_label'] == 'IN-L').astype(int)
label_df['is_mid'] = (label_df['intrinsic_label'] == 'IN-M').astype(int)
label_df['is_small'] = (label_df['intrinsic_label'] == 'IN-S').astype(int)
label_df['is_micro'] = (label_df['intrinsic_label'] == 'IN-Mi').astype(int)

# Calculate correlation and sort features by importance
correlation_features = []
correlation_scores = []

for feature in all_features:
    if feature in label_df.columns:
        # Calculate correlation with each label class
        corr_large = abs(label_df[feature].corr(label_df['is_large']))
        corr_mid = abs(label_df[feature].corr(label_df['is_mid']))
        corr_small = abs(label_df[feature].corr(label_df['is_small']))
        corr_micro = abs(label_df[feature].corr(label_df['is_micro']))
        
        # Use the maximum correlation with any class
        max_corr = max(corr_large, corr_mid, corr_small, corr_micro)
        
        if not np.isnan(max_corr):
            correlation_features.append(feature)
            correlation_scores.append(max_corr)

# Create a DataFrame for visualization
corr_df = pd.DataFrame({'Feature': correlation_features, 'Correlation': correlation_scores})
corr_df = corr_df.sort_values('Correlation', ascending=False)

print("\nTop 15 features by correlation to labels:")
print(corr_df.head(12))

# Select top features based on correlation
top_n_features = 12
selected_features = corr_df.head(top_n_features)['Feature'].tolist()

# Always include market cap numeric as it's a critical feature
if 'market_cap_numeric' not in selected_features:
    selected_features.append('market_cap_numeric')

print(f"\nSelected {len(selected_features)} features for clustering:")
for feature in selected_features:
    print(f"- {feature}")

# Prepare data for clustering
print("\nPreparing data for clustering with selected features...")
X_all = df[selected_features]

# Use RobustScaler to be less affected by outliers
scaler = RobustScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Apply feature weights based on correlation scores
feature_weights = {}
for feature in selected_features:
    if feature in corr_df['Feature'].values:
        weight = float(corr_df[corr_df['Feature'] == feature]['Correlation'].values[0]) * 2
        feature_weights[feature] = max(weight, 0.5)  # Ensure minimum weight of 0.5
    else:
        feature_weights[feature] = 1.0  # Default weight

# Ensure market_cap_numeric has high weight
feature_weights['market_cap_numeric'] = max(feature_weights.get('market_cap_numeric', 0), 3.0)

print("\nFeature weights for clustering:")
for feature, weight in feature_weights.items():
    print(f"{feature}: {weight:.2f}")

# Apply feature weights
weighted_X = np.copy(X_all_scaled)
for i, feature in enumerate(selected_features):
    weighted_X[:, i] *= feature_weights[feature]

# Determine optimal number of clusters using silhouette score
print("\nDetermining optimal number of clusters...")
silhouette_scores = []
k_range = range(2, 7)  # Try 2-6 clusters

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(weighted_X)
    score = silhouette_score(weighted_X, cluster_labels)
    silhouette_scores.append(score)
    print(f"Silhouette score for {k} clusters: {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'o-', markersize=8)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Method For Optimal k')
plt.grid(True)
plt.savefig('optimal_clusters.png')
print("Optimal clusters visualization saved as 'optimal_clusters.png'")

# Use 4 clusters as per business requirement (Large, Mid, Small, Micro)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

# Fit KMeans on weighted features
print("\nPerforming KMeans clustering with weighted features...")
df['cluster'] = kmeans.fit_predict(weighted_X)

# Calculate silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(weighted_X, df['cluster'])
print(f"Final Silhouette Score: {silhouette_avg:.4f}")

# Analyze clusters
print("\nCluster distribution:")
print(df['cluster'].value_counts())

# Calculate average market cap numeric for each cluster
print("\nAverage market cap numeric by cluster:")
for cluster_id in range(n_clusters):
    avg_market_cap = df[df['cluster'] == cluster_id]['market_cap_numeric'].mean()
    print(f"Cluster {cluster_id}: {avg_market_cap:.2f}")

# Map clusters to categories based on average market cap ranking
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

# Check label distribution in each cluster to verify
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

plt.title('PCA Visualization of Stock Clusters (Correlation-Weighted)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.savefig('stock_clusters.png')
print("Cluster visualization saved as 'stock_clusters.png'")

# Train a Random Forest classifier on labeled data to validate approach
print("\nTraining a Random Forest classifier to validate approach...")
labeled_data = df[df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi'])]
X_labeled = labeled_data[selected_features]
y_labeled = labeled_data['intrinsic_label'].map({'IN-L': 0, 'IN-M': 1, 'IN-S': 2, 'IN-Mi': 3})

X_train, X_test, y_train, y_test = train_test_split(
    scaler.transform(X_labeled), y_labeled, test_size=0.2, random_state=42, stratify=y_labeled
)

# Use grid search to find best parameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"\nBest parameters: {grid_search.best_params_}")

# Use best model for final evaluation
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

print("\nRandom Forest classifier performance on labeled data:")
print(classification_report(y_test, y_pred))

# Calculate feature importances
feature_importance = best_rf.feature_importances_
features_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importance})
features_df = features_df.sort_values('Importance', ascending=False)

print("\nRandom Forest feature importance ranking:")
for idx, row in features_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Create an enhanced performance metric using insights from feature importance
print("\nCreating enhanced performance metrics...")
performance_weights = {}
for feature in selected_features:
    if feature in features_df['Feature'].values:
        weight = float(features_df[features_df['Feature'] == feature]['Importance'].values[0])
        performance_weights[feature] = weight
    else:
        performance_weights[feature] = 0.01  # Small default weight

# Normalize weights to sum to 1
total_weight = sum(performance_weights.values())
for feature in performance_weights:
    performance_weights[feature] /= total_weight

# Calculate weighted performance score
df['performance_score'] = 0
for feature in selected_features:
    # Normalize feature values to 0-1 scale
    min_val = df[feature].min()
    max_val = df[feature].max()
    range_val = max_val - min_val
    
    if range_val > 0:
        normalized_feature = (df[feature] - min_val) / range_val
    else:
        normalized_feature = 0.5  # Default value if no variation
    
    # For negatively correlated features like volatility, subtract from 1
    if feature in ['volatility', 'price_volatility', 'debt_to_income']:
        normalized_feature = 1 - normalized_feature
    
    # Add weighted contribution to performance score
    df['performance_score'] += normalized_feature * performance_weights.get(feature, 0)

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

# Generate correlation heatmap for selected features
plt.figure(figsize=(16, 14))
corr_matrix = df[selected_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Selected Features')
plt.tight_layout()
plt.savefig('feature_correlation_heatmap.png')
print("Feature correlation heatmap saved as 'feature_correlation_heatmap.png'")

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

# Generate enhanced report
print("\nCreating detailed report...")
with open('stock_selection_report.txt', 'w') as f:
    f.write("NSE STOCK SELECTION REPORT (CORRELATION-BASED CLUSTERING)\n")
    f.write("================================================\n\n")
    
    f.write("DATA SUMMARY\n")
    f.write("-----------\n")
    f.write(f"Total companies: {len(df)}\n")
    f.write(f"Labeled companies: {sum(df['intrinsic_label'].isin(['IN-L', 'IN-M', 'IN-S', 'IN-Mi']))}\n\n")
    
    f.write("FEATURE SELECTION\n")
    f.write("----------------\n")
    f.write("Top 15 features by correlation to market cap labels:\n")
    for idx, row in corr_df.head(15).iterrows():
        f.write(f"- {row['Feature']}: {row['Correlation']:.4f}\n")
    f.write("\n")
    
    f.write("RANDOM FOREST FEATURE IMPORTANCE\n")
    f.write("-----------------------------\n")
    for idx, row in features_df.iterrows():
        f.write(f"- {row['Feature']}: {row['Importance']:.4f}\n")
    f.write("\n")
    
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
        f.write("-" * (len(nin_label) + 19) + "\n")
        for i in range(len(companies)):
            row = companies.iloc[i]
            f.write(f"{i+1}. {row['symbol']} - {row['company_name']}\n")
            f.write(f"   Performance Score: {row['performance_score']:.4f}\n")
            f.write(f"   Market Cap Category: {row['market_cap_category']}\n")
            
            # Add key financial metrics
            if 'sharpe_ratio' in row:
                f.write(f"   Sharpe Ratio: {row['sharpe_ratio']:.4f}\n")
            if 'profit_margin' in row:
                f.write(f"   Profit Margin: {row['profit_margin']:.4f}\n")
            if 'return_on_equity' in row:
                f.write(f"   Return on Equity: {row['return_on_equity']:.4f}\n")
            if 'year_change_percent' in row:
                f.write(f"   Year Change %: {row['year_change_percent']:.2f}%\n")
            f.write("\n")

print("\nAnalysis complete! You now have:")
print("1. Updated CSV with new labels for high-performing companies")
print("2. Multiple visualizations showing clustering and feature relationships")
print("3. A detailed report of all selected companies with their performance metrics")
print("4. Feature importance analysis from both correlation and Random Forest")