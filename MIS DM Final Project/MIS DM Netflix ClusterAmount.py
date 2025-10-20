#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 11:49:43 2025

@author: joelforson
"""
pip in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler #when the distribution is normal
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder



df_ = pd.read_csv('/Users/joelforson/Downloads/Final_Cleaned_Netflix_Dataset (1).csv')
df_.isnull().sum()
df_=df_.dropna()
df = df_[['release_year',
       'Available Globally?', 'Release Date', 'Hours Viewed',
       'Avg Runtime (mins)', 'Views', 'Seasons', 'id', 'type', 'description',
       'age_certification', 'runtime', 'genres', 'production_countries',
       'seasons', 'imdb_score', 'imdb_votes', 'tmdb_popularity', 'tmdb_score']]

df['genres'] = df['genres'].str.replace(r"[\[\]']", "", regex=True)
df['genres'] = df['genres'].str.split(',')
df = df.explode('genres')
df['genres'] = df['genres'].str.strip()
# List of categorical columns you want to encode
cat_cols = ['type', 'age_certification', 'genres']
le = LabelEncoder()

# Apply label encoding to each categorical column
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

def convert_time_to_minutes(time_str):
    if pd.isnull(time_str):
        return None
    parts = str(time_str).split(':')
    if len(parts) == 2:
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes
    elif len(parts) == 1:  # already in minutes
        return int(parts[0])
    else:
        return None  # fallback for bad data

#df['Avg Runtime (mins)'] = df['Avg Runtime (mins)'].apply(convert_time_to_minutes)

df_=pd.get_dummies(df_,drop_first=True)
df =pd.get_dummies(df,drop_first=True)
df = df.dropna()

scaler = StandardScaler()
scaler.fit(df)
scaled_df=scaler.transform(df)
scaled_df


# Part 1
wcv=[]
silk_score=[]
for i in range(2,15):
    km=KMeans(n_clusters=i,random_state=0)   #pre-specifying K
    km.fit(scaled_df) #training/finding clusters
    wcv.append(km.inertia_) #add within cluster variation to wcv list
    silk_score.append(silhouette_score(scaled_df, km.labels_))

    
#plotting wcv with # of clusters

plt.plot(range(2,15),wcv)
plt.xlabel("Number of Cluster")
plt.ylabel("Within Cluster Variation")
plt.show()

##plotting the sillhoutte distance

plt.plot(range(2,15),silk_score)
plt.xlabel("Number of Cluster")
plt.ylabel("Silhoutte score")
plt.grid()
plt.show()

#Use the scaled data to apply K-means. Using the 
#elbow method and silhouette score to identify the n
#number of clusters,justify your choice.
#the elbow seems to be somewhere between 3-6 relatively high silhoutte score at 3 as well
#going with no of clusters=3
kmeans=KMeans(n_clusters=7,random_state=0)
kmeans.fit(scaled_df)

#adding labels to data
df['labels']=kmeans.labels_

#interpret the clusters
kmmm=df.groupby('labels').mean()
#df_['labels']=kmeans.labels_

##visualize using bar plot
plt.figure(figsize=(200,200))
kmmm.plot(kind='bar')
plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right')
plt.show()


#Q2
#Perform hierarchical clustering using the complete linkage method. Plot the dendrogram and
#suggest an appropriate cutoff to determine the number of clusters.
plt.figure(figsize=(10,7))
plt.axhline(y=50, color='r', linestyle='--')#(plots the horizontal line)you will not need to show this line
dendrogram(linkage(scaled_df,method='ward'))
plt.show()

#complete linkage with 6 clusters
hc=AgglomerativeClustering(n_clusters=6,linkage='complete')
hc.fit(scaled_df)


#add labels to df
df['labels']=hc.labels_

hc_mean = df.groupby('labels').mean()
##visualize using bar plot
plt.figure(figsize=(10,10))
hc_mean.plot(kind='bar')
plt.legend(bbox_to_anchor=(1.5, 1),loc='upper right')
plt.show()


# How many components did you retain?
from sklearn.decomposition import PCA
pca=PCA(n_components=6)
pca.fit(scaled_df)
pca_df=pca.transform(scaled_df)
pca_df=pd.DataFrame(pca_df,columns=['PC1','PC2','PC3','PC4','PC5','PC6'])

##how much variance was captured
#2 or more principal components can capture atleas
#t 80% variance
pca.explained_variance_ratio_

#Interpret the first principal component. Which or
#iginal variables contribute the most, and what does
# this component represent? Use a threshold value of 0.4.
pca_df.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)
total_set=pd.concat([pca_df,df],axis=1)
loadings=total_set.corr()
loadings
##from the loadings we can see that PC1 is highly
#correlated with Apps (applications received),Ac
#cept (applications accepted
#and Enroll (students enrolled)
#PC1 can be thought of a a variable that tells how
#many students attend the university and how many
#people applied.
