import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import joblib

try:
    df = pd.read_csv("data/Mall_Customers.csv")
except FileNotFoundError:
    print("Error: Mall_Customers.csv not found. Please make sure the file is in the correct directory.")
    exit()

X1 = df[["Annual Income (k$)", "Spending Score (1-100)"]].copy()

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(X1)

ks1 = range(2, 9)
sils1 = []
for k in ks1:
    km1 = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels1 = km1.fit_predict(X1_scaled)
    sils1.append(silhouette_score(X1_scaled, labels1))

best_k1 = ks1[np.argmax(sils1)]
max_sil1 = max(sils1)
print(f"Best k for 2 features: {best_k1} (Silhouette Score: {max_sil1:.4f})")

kmeans1 = KMeans(n_clusters=best_k1, n_init=10, random_state=42)
df["Cluster_2_Features"] = kmeans1.fit_predict(X1_scaled)

joblib.dump(kmeans1, 'models/kmeans_2_features.joblib')
joblib.dump(scaler1, 'modes/scaler_2_features.joblib')

centroids1 = scaler1.inverse_transform(kmeans1.cluster_centers_)
plt.figure(figsize=(10, 7))
scatter1 = plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster_2_Features"], cmap='viridis', alpha=0.7, label='Customers')
plt.scatter(centroids1[:, 0], centroids1[:, 1], marker="X", s=200, c='red', label='Centroids')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segments (2 Features, k={best_k1})")
plt.legend()
plt.grid(True)
plt.show()

print(df.groupby("Cluster_2_Features")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))

X2 = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].copy()

scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(X2)

ks2 = range(2, 9)
sils2 = []
for k in ks2:
    km2 = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels2 = km2.fit_predict(X2_scaled)
    sils2.append(silhouette_score(X2_scaled, labels2))

best_k2 = ks2[np.argmax(sils2)]
max_sil2 = max(sils2)
print(f"Best k for 3 features: {best_k2} (Silhouette Score: {max_sil2:.4f})")

kmeans2 = KMeans(n_clusters=best_k2, n_init=10, random_state=42)
df["Cluster_3_Features"] = kmeans2.fit_predict(X2_scaled)

joblib.dump(kmeans2, 'models/kmeans_3_features.joblib')
joblib.dump(scaler2, 'models/scaler_3_features.joblib')

centroids2 = scaler2.inverse_transform(kmeans2.cluster_centers_)
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
scatter2 = ax.scatter(df["Age"], df["Annual Income (k$)"], df["Spending Score (1-100)"], c=df["Cluster_3_Features"], cmap='plasma', alpha=0.7, label='Customers')
ax.scatter(centroids2[:, 0], centroids2[:, 1], centroids2[:, 2], marker="X", s=250, c='red', label='Centroids')
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
ax.set_title(f"Customer Segments (3 Features, k={best_k2})")
plt.legend()
plt.show()

print(df.groupby("Cluster_3_Features")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))

print("\n\n--- Comparison of Models ---")
print(f"2-Feature Model (Income, Spending):")
print(f"  - Best k: {best_k1}")
print(f"  - Max Silhouette Score: {max_sil1:.4f}\n")

print(f"3-Feature Model (Age, Income, Spending):")
print(f"  - Best k: {best_k2}")
print(f"  - Max Silhouette Score: {max_sil2:.4f}\n")

if max_sil2 > max_sil1:
    print("Conclusion: The 3-feature model has a higher silhouette score, suggesting that its clusters are better defined.")
    print("Adding 'Age' helped create more distinct customer segments.")
else:
    print("Conclusion: The 2-feature model has a higher or equal silhouette score, suggesting that adding 'Age' did not improve cluster definition.")
