# ============================================
# Customer Segmentation using K-Means Clustering
# ============================================

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load Dataset
df = pd.read_csv("mall_customers.csv")

print("First 5 Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

# Step 3: Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 4: Standardize the Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Elbow Method
wcss = []

for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)
    wcss.append(model.inertia_)

# Step 6: Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)

plt.show()

# ---------------------------------------------
# Select K = 5
# ---------------------------------------------

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Step 7: Add Cluster Column
df['Cluster'] = clusters

print("\nDataset with Cluster Labels:")
print(df.head())

# Step 8: Display Cluster Centres
centers = scaler.inverse_transform(kmeans.cluster_centers_)

print("\nCluster Centres")

for i, center in enumerate(centers):
    print(f"Cluster {i}")
    print(f"Annual Income : {center[0]:.2f}")
    print(f"Spending Score: {center[1]:.2f}")
    print()

# Step 9: Scatter Plot
plt.figure(figsize=(10,6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='rainbow',
    s=80
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.show()

# Step 10: Scatter Plot with Centroids
plt.figure(figsize=(10,6))

plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'],
    cmap='rainbow',
    s=80,
    label='Customers'
)

plt.scatter(
    centers[:,0],
    centers[:,1],
    c='black',
    marker='X',
    s=250,
    label='Centroids'
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation with Centroids")

plt.legend()

plt.show()

# Step 11: Display Final Dataset
print("\nFinal Dataset")
print(df)