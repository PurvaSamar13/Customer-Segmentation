import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.shape)
print(data.info())

x = data.iloc[:,[3,4]].values
print(x)
# finding wcss value for different number of clusters
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=5)
data['Cluster'] = kmeans.fit_predict(data[['Spending Score (1-100)']])

plt.scatter(data['CustomerID'], data['Spending Score (1-100)'], c=data['Cluster'])
plt.xlabel('CustomerID')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()

import plotly.express as px
fig = px.scatter(data, x='CustomerID', y='Spending Score (1-100)', color='Cluster', title='Customer Segmentation',
                 labels={'CustomerID':'Customer ID'})
fig.show()

def get_cluster(customer_id, data):
    cluster = data[data['CustomerID']==customer_id]['Cluster'].values[0]
    return cluster

from sklearn.metrics import silhouette_score

kmeans_silhouette = silhouette_score(data[['Spending Score (1-100)']], data['Cluster'])
print(f'K-Means Silhouette Score: {kmeans_silhouette}')

