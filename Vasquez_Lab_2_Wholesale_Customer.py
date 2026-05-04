#Name: Angel Vasquez
#Lab Partner: Ethan Yang 

# Import libraries for data handling, math operations, plotting, scaling, and clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load wholesale customer dataset into dataframe
df = pd.read_csv('Wholesale_customer.csv')

# Print dataset so we can visually inspect it
print(df)

# Check if any columns contain missing values
print(df.isnull().sum())

# Create scatter plot of Fresh vs Frozen spending
# Each dot = one customer
plt.scatter(df['Fresh'],df['Frozen'], c='red', s=2)
plt.xlabel("Fresh")      # Money spent on fresh food
plt.ylabel("Frozen")     # Money spent on frozen food
plt.show()

# SCALE FEATURES
# Convert Fresh and Frozen columns into numeric array
X = np.array(list(zip(df['Fresh'], df['Frozen'])))

# Create scaler object
scaler = StandardScaler()

# Scale the data so both features are on same scale
# This prevents large dollar values from dominating clustering
X = scaler.fit_transform(X) 

# Now X contains scaled values instead of raw dollar amounts


# -------- K-MEANS CLUSTERING --------
# Choose number of clusters (k = 4 customer groups)
k=4

# Create K-Means model
# n_init=10 means algorithm tries 10 different starting points and picks best result
model=KMeans(n_clusters=k, n_init=10)

# Train model using scaled data
kmeans=model.fit(X)

# Assign each customer to a cluster
labels=model.predict(X)

# Get center point (average customer behavior) for each cluster
centroids=model.cluster_centers_

# Print cluster assignment for each customer
print(labels)

# Print centroid locations (average spending pattern for each group)
print(centroids)


# PLOT CLUSTERS 
# Assign colors to clusters for visualization
c=['m','g','b','r','y','c']
colors=[c[i] for i in labels]

# Plot scaled customer data
plt.scatter(X[:, 0], X[:, 1], c=colors, s=2)

# Plot cluster centers using black star markers
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=100, c='black')

plt.xlabel("Fresh (Scaled)")
plt.ylabel("Frozen (Scaled)")
plt.show()


# PREDICT NEW CUSTOMER CLUSTER
# New customer example: $12,000 Fresh and $3,000 Frozen spending
# Must scale new data using SAME scaler
scaled_pt = scaler.transform([[12000, 3000]])

# Predict cluster for new customer
cluster=model.predict(scaled_pt)[0]

# Print color representing cluster
print(c[cluster])


#SILHOUETTE SCORE
#Used to measure how good cluster separation is
from sklearn.metrics import silhouette_score

# Test cluster quality when k = 2
model=KMeans(n_clusters=2, n_init=10)
model.fit(X)
label=model.predict(X)
print(f'silhouette Score (n=2): {silhouette_score(X,label)}')

# Test cluster quality when k = 3
model=KMeans(n_clusters=3, n_init=10)
model.fit(X)
label=model.predict(X)
print(f'silhouette Score (n=3): {silhouette_score(X,label)}')

# Test cluster quality when k = 4
model=KMeans(n_clusters=4, n_init=10)
model.fit(X)
label=model.predict(X)
print(f'silhouette Score (n=4): {silhouette_score(X,label)}')


#FINAL CLUSTER MODEL
# Run final clustering using chosen k value
k=4

# Train final K-Means model
kmeans=KMeans(n_clusters=k, n_init=10)
kmeans=kmeans.fit(X)

# Get final cluster assignments and centers
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

# Assign final colors to clusters
c=['b','r','y','g','c','m']
colors=[c[i] for i in labels]

# Plot final clusters
plt.scatter(X[:,0], X[:,1], c=colors, s=20)

# Plot final centroid locations
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=100, c='black')

plt.xlabel("Fresh (Scaled)")
plt.ylabel("Frozen (Scaled)")
plt.show()

# Print final centroid values
print(centroids)
