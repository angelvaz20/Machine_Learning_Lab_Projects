#Call the libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df=pd.read_csv("bermuda_shorts.csv")
print(df.head())

print(df.info())

print(df.shape)

df.isnull().sum()

#remove the missing values with NaNs
df=df.dropna(subset=['bmxleg','bmxwaist'])
print(df.shape)

#scatter plot
plt.scatter(df['bmxleg'],df['bmxwaist'], c='red', s=2)
plt.xlabel("upper leg length (cm)")
plt.ylabel("waist circumference (cm)")

print(plt.show())

#For now, we'll try k value 2
k=2
X=np.array(list(zip(df['bmxleg'], df['bmxwaist'])))
model=KMeans(n_clusters=k)
kmeans=model.fit(X)
labels=model.predict(X)
centroids=model.cluster_centers_

print(labels)
print(centroids)


#Plot the data and centroids on a scatter plot
#map the labels to colors

c=['m','g','b','r','y','c']
colors=[c[i] for i in labels]

plt.scatter(df['bmxleg'], df['bmxwaist'], c=colors, s=2)
plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=100, c='black')
print(plt.show())


#Using the model that you have just trained, 
#you can use it to predict what cluster for any data pt

cluster=model.predict([[35,123]])[0]
print(c[cluster])


#To determine the silhouette coefficient/score

from sklearn.metrics import silhouette_score

model=KMeans(n_clusters=2)
model.fit(X)
label=model.predict(X)

print(f'silhouette Score (n=2): {silhouette_score(X,label)}')


model=KMeans(n_clusters=3)
model.fit(X)
label=model.predict(X)

print(f'silhouette Score (n=3): {silhouette_score(X,label)}')


model=KMeans(n_clusters=4)
model.fit(X)
label=model.predict(X)

print(f'silhouette Score (n=4): {silhouette_score(X,label)}')



k=4
X=np.array(list(zip(df['bmxleg'],df['bmxwaist'])))

kmeans=KMeans(n_clusters=k)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
centroids=kmeans.cluster_centers_

#map the labels to colors
c=['b','r','y','g','c','m']
colors=[c[i] for i in labels]

plt.scatter(df['bmxleg'],df['bmxwaist'],c=colors,s=2)
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=100,c='black')

print(plt.show())

print(centroids)


