import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm

def wcss(df):
    wcss_arr = [] 
    for i in range(1, 11): 
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 15)
        kmeans.fit(df) 
        wcss_arr.append(kmeans.inertia_)


def wcss_plot(df, title):
    wcss_arr = [] 
    for i in range(1, 11): 
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 15)
        kmeans.fit(df) 
        wcss_arr.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot([i for i in range(1,11)] ,wcss_arr)
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title(title)
    
