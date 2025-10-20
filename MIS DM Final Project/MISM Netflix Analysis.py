#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 00:41:59 2025

@author: joelforson
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/joelforson/Downloads/Final_Cleaned_Netflix_Dataset (1).csv')

# Take a quick look at the columns
print("DataFrame Info:")
print(df.info())
print("\nSample Rows:")
print(df.head())


#Do some exploratory Data Analysis

#Data Preprocessing: Encoding for genres, countries, and type

# == 2.1 One-hot encoding for genres ==
if "genres" in df.columns:
    # Split genre strings into a list
    df["genres_list"] = df["genres"].apply(
        lambda x: [g.strip() for g in x.split(",")] if isinstance(x, str) else []
    )
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(df["genres_list"])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb_genres.classes_, index=df.index)
else:
    genres_df = pd.DataFrame()

# == 2.2 One-hot encoding for production countries ==
if "countries" in df.columns:
    df["countries_list"] = df["countries"].apply(
        lambda x: [c.strip() for c in x.split(",")] if isinstance(x, str) else []
    )
    mlb_countries = MultiLabelBinarizer()
    countries_encoded = mlb_countries.fit_transform(df["countries_list"])
    countries_df = pd.DataFrame(countries_encoded, columns=mlb_countries.classes_, index=df.index)
else:
    print("No 'countries' column found. Skipping country encoding...")
    countries_df = pd.DataFrame()

# == 2.3 One-hot encoding for type (Movie vs TV Show) ==
if "type" in df.columns:
    type_encoded = pd.get_dummies(df["type"], prefix="type")
else:
    print("No 'type' column found. Skipping type encoding...")
    type_encoded = pd.DataFrame()

# Combine everything into a single feature set X
X = pd.concat([genres_df, countries_df, type_encoded], axis=1)

print("\nShape of one-hot-encoded features (X):", X.shape)
print("Columns in X:", X.columns.tolist())


#K-Means Clustering

from sklearn.cluster import KMeans

k = 8
kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# Add cluster labels back to original df
df["cluster"] = cluster_labels
print("\nCluster label counts:")
print(df["cluster"].value_counts())


#Average Performance Metrics by Cluster

for col in ["Views", "Hours Viewed", "imdb_score"]:
    if col not in df.columns:
        print(f"Warning: column '{col}' not found in df. Check your dataset columns.")

# Using Engagement Metrics; "Views", "Hours Viewed", "imdb_score"
metric_cols = ["Views", "Hours Viewed", "imdb_score"]
existing_metrics = [m for m in metric_cols if m in df.columns]

cluster_performance = df.groupby("cluster")[existing_metrics].mean().round(2)

print("\nAverage performance metrics by cluster:")
print(cluster_performance)


#PCA for Visualization

from sklearn.decomposition import PCA

# Reduce to 2 components
pca = PCA(n_components=2, random_state=1)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
pca_df["cluster"] = df["cluster"]

# Plot the clusters 
plt.figure(figsize=(8, 6))
for clust in sorted(pca_df["cluster"].unique()):
    subset = pca_df[pca_df["cluster"] == clust]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Cluster {clust}", alpha=0.5)

plt.title("K-means Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()


#Bar charts to measure engagement data

#Creating separate plots for each metric 
for metric in existing_metrics:
    plt.figure()
    plt.bar(cluster_performance.index.astype(str), cluster_performance[metric])
    plt.title(f"Average {metric.capitalize()} by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel(f"Average {metric.capitalize()}")
    plt.show()


#Cluster Profiles for Categorical data

if not genres_df.empty:
    genres_by_cluster = pd.concat([genres_df, df["cluster"]], axis=1).groupby("cluster").mean()
    # Show top 5 genres for each cluster
    for clust in range(k):
        row = genres_by_cluster.loc[clust]
        top_genres = row.sort_values(ascending=False).head(5)
        print(f"\nCluster {clust} - Top 5 Genres:\n{top_genres}")

if not countries_df.empty:
    countries_by_cluster = pd.concat([countries_df, df["cluster"]], axis=1).groupby("cluster").mean()
    # Show top 5 countries for each cluster
    for clust in range(k):
        row = countries_by_cluster.loc[clust]
        top_countries = row.sort_values(ascending=False).head(5)
        print(f"\nCluster {clust} - Top 5 Countries:\n{top_countries}")

if not type_encoded.empty:
    type_by_cluster = pd.concat([type_encoded, df["cluster"]], axis=1).groupby("cluster").mean()
    # Show average distribution of Movie vs Show
    for clust in range(k):
        row = type_by_cluster.loc[clust]
        print(f"\nCluster {clust} - Movie vs Show (averages):\n{row}")
