#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# In[176]:


#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[177]:


#data load
dataset = pd.read_csv(r'D:\ML&AI\Batch1\04.Machine Learning\05.Unsupervised Learning\Classes\Mall_Customers_to1qc.csv')


# In[178]:


dataset.head()


# In[179]:


dataset.info()


# In[180]:


dataset.describe()


# In[181]:


dataset.hist(figsize=(11,8))
plt.show()


# In[182]:


# Just considering required features
X = dataset.iloc[:,3:]


# In[183]:


X


# In[184]:


wcss = [] # within cluster sum of squares


# In[185]:


# Finding best values for K. b/w 1 - 10
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #within cluster sum of squares


# In[186]:


wcss


# In[187]:


plt.plot(range(1,11), wcss)
plt.grid()
plt.show()


# In[188]:


#Considered K=6 as its best fit - there is no significant change in the wcss or inertia
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42)


# In[189]:


# predicted clasifications/clusters
y_kmeans = kmeans.fit_predict(X)


# In[190]:


# 6 clusters for all rows 
y_kmeans


# In[191]:


X.columns


# In[192]:


plt.scatter('Annual Income (k$)', 'Spending Score (1-100)')
plt.show()


# In[193]:


#Ploting to see the clusters predicted
plt.figure(figsize=(13,11))
plt.scatter(X[y_kmeans==0]['Annual Income (k$)'],X[y_kmeans==0]['Spending Score (1-100)'],s=100,c='red',label='C1')
plt.scatter(X[y_kmeans==1]['Annual Income (k$)'],X[y_kmeans==1]['Spending Score (1-100)'],s=100,c='blue',label='C2')
plt.scatter(X[y_kmeans==2]['Annual Income (k$)'],X[y_kmeans==2]['Spending Score (1-100)'],s=100,c='green',label='C3')
plt.scatter(X[y_kmeans==3]['Annual Income (k$)'],X[y_kmeans==3]['Spending Score (1-100)'],s=100,c='cyan',label='C4')
plt.scatter(X[y_kmeans==4]['Annual Income (k$)'],X[y_kmeans==4]['Spending Score (1-100)'],s=100,c='magenta',label='C5')
plt.scatter(X[y_kmeans==5]['Annual Income (k$)'],X[y_kmeans==5]['Spending Score (1-100)'],s=100,c='black',label='C6')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# ## End  
