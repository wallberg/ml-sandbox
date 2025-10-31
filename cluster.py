import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

N_CLUSTERS = 4

input_file = "1_sampled.csv"
output_file = "2_clustered.csv"

df = pd.read_csv(input_file)

feature_cols = ['duration_seconds', 'content_length_tokens']
x = df[feature_cols].fillna(0)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
kmeans_labels = kmeans.fit_predict(x_scaled)
df['kmeans_content_duration_clusters'] = kmeans_labels

dbscan = DBSCAN(eps=0.1, min_samples=10)
dbscan_labels = dbscan.fit_predict(x_scaled)
df['dbscan_content_duration_clusters'] = dbscan_labels

#plotting Kmeans
markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']

plt.figure(figsize=(8,16))
for cluster_id in sorted(df['kmeans_content_duration_clusters']):
    subset = df[df['kmeans_content_duration_clusters'] == cluster_id]
    plt.scatter(
        subset['duration_seconds'],
        subset['content_length_tokens'],
        label=f'Cluster {cluster_id}',
        edgecolors='black',
        linewidths=0.5,
        marker=markers[cluster_id % len(markers)],
    )

plt.xlabel("duration_seconds")
plt.ylabel("content_length_tokens")
plt.title("KMeans Clustering of Content Duration and Length")

plt.savefig("kmeans_plot_file.png")
plt.close()

#plotting DBSCAN
plt.figure(figsize=(8,16))
for cluster_id in sorted(df['dbscan_content_duration_clusters']):
    subset = df[df['dbscan_content_duration_clusters'] == cluster_id]
    plt.scatter(
        subset['duration_seconds'],
        subset['content_length_tokens'],
        label=f'Cluster {cluster_id}',
        edgecolors='black',
        linewidths=0.5,
    )

plt.xlabel("duration_seconds")
plt.ylabel("content_length_tokens")
plt.title("DBSCAN Clustering of Content Duration and Length")

plt.savefig("dbscan_plot_file.png")
plt.close()

# sampled_df.to_csv(output_file, index=False)
