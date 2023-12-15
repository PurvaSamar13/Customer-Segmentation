import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv("C:/Users/Purva samar/PycharmProjects/pythonProject2/Mall_Customers.csv")
print(customer_data.head())
print(customer_data.shape)
print(customer_data.info())

# iloc:integer location ,":"before the comma represents all rows in the DataFrame.
x = customer_data.iloc[:,[3,4]].values
print(x)
# findinf wcss value for different number of clusters
wcss = []
''' "init" is a parameter defines the method for initializing the centroids before clustering begins
 "k-means++ is a smart initialization method that tries to place centroids in a way that speeds up convergence
 Setting random_state to a specific value (e.g., 42 in this case) ensures that if you run the code multiple times, you will get the same results
 The "fit()" method is used to fit the KMeans model to the data "x".'''
''''"kmeans.inertia_" returns the WCSS (inertia), which is a measure of how internally coherent the clusters are.
This value represents the sum of squared distances of samples to their closest cluster center.'''
'''for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('wcss')
plt.show()'''

# Training the K-means clustering model (i=5)
# "fit_predict" computes the centroids of clusters based on the input data
'''kmeans = KMeans(n_clusters=5,init='k-means++', random_state=0)
y = kmeans.fit_predict(x)
print(y)'''

plt.figure(figsize=(8,8))
# Plotting the clusters
# "s=50" Sets the size of the plotted points to 50
'''plt.scatter(x[y==0,0],x[y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(x[y==1,0],x[y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(x[y==2,0],x[y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(x[y==3,0],x[y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(x[y==4,0],x[y==4,1],s=50,c='blue',label='Cluster 5')'''

# Plotting the centroids
'''kmeans.cluster_centers_[:,0] represents the x axis value of centroids
kmeans.cluster_centers_[:,1] represents the y axis value of centroids'''
'''plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()'''





