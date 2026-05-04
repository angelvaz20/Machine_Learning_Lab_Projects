# Name: Angel Vasquez
# Lab Partner: Ethan Yang
# Import libraries used for math, data handling, plotting, and clustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load the Google review dataset into a dataframe
df=pd.read_csv("google_review_ratings.csv")

# Show first few rows so we can see what the data looks like
print(df.head())

# Show column names, data types, and non-null counts
print(df.info())

# Show number of rows and columns
print(df.shape)

# Check how many missing values exist in each column
df.isnull().sum()

# Remove rows where pubs/bars or juice bars ratings are missing
df=df.dropna(subset=['Category 10','Category 14'])
print(df.shape)


# Create a basic scatter plot of pubs/bars vs juice bars ratings
# Each dot represents one location
plt.scatter(df['Category 10'],df['Category 14'], c='red', s=2)
plt.xlabel("pubs/bars")
plt.ylabel("juice bars")

print(plt.show())


# Choose number of clusters (k = 2 means 2 groups will be created)
k=2

# Create X dataset using only the two features we want to cluster
# This converts the two columns into a numeric array for K-Means
X=np.array(list(zip(df['Category 10'], df['Category 14'])))

# Create K-Means model
model=KMeans(n_clusters=k)

# Train the model using the data
kmeans=model.fit(X)

# Assign each data point to a cluster
labels=model.predict(X)

# Get the center point of each cluster
centroids=model.cluster_centers_

# Print cluster assignment for each point
print(labels)

# Print centroid coordinates
print(centroids)


# Assign colors to clusters so they are easy to see on plot
c=['m','g','b','r','y','c']
colors=[c[i] for i in labels]

# Plot clustered data points with colors
plt.scatter(df['Category 10'], df['Category 14'], c=colors, s=20)

# Plot centroid locations using black star markers
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=200, c='black')

plt.show()


# Predict which cluster a new data point belongs to
# Example: location with rating 3.5 pubs and 4.2 juice bars
cluster=model.predict([[3.5, 4.2]])[0]

# Print the color representing that cluster
print(c[cluster])


# Import silhouette score tool to evaluate clustering quality
from sklearn.metrics import silhouette_score


# Test clustering quality when k = 2
model=KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(X)
label=model.predict(X)

# Higher silhouette score = better cluster separation
print(f'silhouette Score (n=2): {silhouette_score(X,label)}')


# Test clustering quality when k = 3
model=KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)
label=model.predict(X)

print(f'silhouette Score (n=3): {silhouette_score(X,label)}')


# Test clustering quality when k = 4
model=KMeans(n_clusters=4, random_state=42, n_init=10)
model.fit(X)
label=model.predict(X)

print(f'silhouette Score (n=4): {silhouette_score(X,label)}')


# Final clustering using chosen k value (k = 4 here)
k=4

# Rebuild X dataset (same two features)
X=np.array(list(zip(df['Category 10'],df['Category 14'])))

# Train final K-Means model with stable random seed
kmeans=KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans=kmeans.fit(X)

# Get cluster labels and centroids
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

# Assign final cluster colors
c=['b','r','y','g','c','m']
colors=[c[i] for i in labels]

# Plot final clustered data
plt.scatter(df['Category 10'],df['Category 14'],c=colors,s=20)

# Plot final centroid locations
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='black')

print(plt.show())

# Print final centroid coordinates
print(centroids)
