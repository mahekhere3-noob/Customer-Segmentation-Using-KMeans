import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load Dataset
# -------------------------

df = pd.read_csv("movies.csv")

print(df.head())

# -------------------------
# Features
# -------------------------

X = df[['Action',
        'Comedy',
        'Drama',
        'Horror',
        'Romance',
        'SciFi']]

# -------------------------
# Standardization
# -------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------
# Elbow Method
# -------------------------

wcss=[]

for i in range(1,8):

    model=KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.plot(range(1,8),wcss,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# -------------------------
# Train Model
# -------------------------

kmeans=KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df['Cluster']=kmeans.fit_predict(X_scaled)

print(df)

# -------------------------
# Cluster Centres
# -------------------------

centers=scaler.inverse_transform(
    kmeans.cluster_centers_
)

centers=pd.DataFrame(
    centers,
    columns=X.columns
)

print("\nCluster Preferences")
print(centers)

# -------------------------
# Recommend Movies
# -------------------------

user=5

cluster=df.loc[df['UserID']==user,'Cluster'].values[0]

print("\nUser",user,"belongs to Cluster",cluster)

recommend=df[df['Cluster']==cluster]

print("\nSimilar Users")

print(recommend[['UserID',
                 'Action',
                 'Comedy',
                 'Drama',
                 'Romance',
                 'SciFi']])