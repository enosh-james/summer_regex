'''

import pandas as pd
import numpy as np

#df =  pd.read_csv('D:\\enosh_regex\\Datasets\\mall.csv')
#print(df.head())

df=df.drop(columns=['CustomerID','Genre'])
#print(df.head())

x=df.iloc[: , [0,1]].values

from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt
a = []


for i in range(1,11):
 b=KMeans(n_clusters= i, init = "k-means++" , random_state = 42)
 b.fit(x)
 a.append(b.inertia_)


plt.plot(range(1,11) , a)

plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list') 
#print(plt.show())



b=KMeans(n_clusters= i, init = "k-means++" , random_state = 42)
y_predict= b.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_predict == 0,0], x[y_predict == 0,1], s=100, c='blue', label = 'Cluster2')
plt.scatter(x[y_predict == 1,0], x[y_predict == 1,1], s=100, c='green', label = 'Cluster2')
plt.scatter(x[y_predict == 2,0], x[y_predict == 2,1], s=100, c='red', label = 'Cluster3')
plt.scatter(x[y_predict == 3,0], x[y_predict == 3,1], s=100, c='cyan', label = 'Cluster4')


plt.scatter(b.cluster_centers_[:,0],b.cluster_centers_[:,1], s=300, c='yellow', label = 'Centroid')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
print(plt.show())
'''

# PCA : PRINCIPLE COMPONENT ANALYSIS ---> changes data from higher dimension to lower dimension without changing its # perfomance.

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df=pd.read_csv('D:\\enosh_regex\\Datasets\\Social_Network_Ads.csv')
#print(df.head())

df = df.drop(columns = ['User ID', 'Gender'])
x = df.drop(columns = ['Purchased'], axis = 1)
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

pipe = Pipeline([('scaler',StandardScaler()),('pca', PCA(n_components=2)),('classifier',RandomForestClassifier(n_estimators=100,random_state=42))])

Pipeline(steps=[('scaler',StandardScaler()),('pca', PCA(n_components=2)),('classifier',RandomForestClassifier(n_estimators=100,random_state=42))])

pipe.fit(x_train,y_train)

Pipeline([('scaler',StandardScaler()),('pca', PCA(n_components=2)),('classifier',RandomForestClassifier(n_estimators=100,random_state=42))])

y_pred = pipe.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print(acc)










































































































