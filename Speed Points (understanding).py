import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import wcss_plot

## Importing data 
data = pd.read_csv('questionnaire response.csv')
df = data.dropna()
column_names = list(df.columns.values)


# Obtain column index
slow_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝘀𝗹𝗼𝘄 in an urban city setting?")
moderate_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝗺𝗼𝗱𝗲𝗿𝗮𝘁𝗲 in an urban city setting?")
fast_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝗳𝗮𝘀𝘁 in an urban city setting?")



### Clustering by speed points
## Slow
# Obtaining data points 
slow_data = df.iloc[:, slow_col].to_numpy()
slow_data_min = np.array([int(s.split("-")[0]) for s in slow_data]).flatten()
slow_data_max = np.array([int(s.split("-")[1]) for s in slow_data]).flatten()
slow_df = pd.DataFrame({'min':slow_data_min, 'max':slow_data_max})

# WCSS and Elbow Method
wcss_plot(slow_df, "Slow Speed Elbow Graph")

# Clustering with optimal clusters
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 15)
kmeans.fit(slow_df)
y_km = kmeans.fit_predict(slow_df)

plt.figure()
plt.scatter(slow_df.loc[:,"min"], slow_df.loc[:,"max"], c=y_km, alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k')
plt.xlabel("Min Speed (in km/h)")
plt.ylabel("Max Speed (in km/h)")
plt.title("Perceived Slow Speed")


## Moderate 
# Obtaining data points 
moderate_data = df.iloc[:, moderate_col].to_numpy()
moderate_data_min = np.array([int(s.split("-")[0]) for s in moderate_data]).flatten()
moderate_data_max = np.array([int(s.split("-")[1]) for s in moderate_data]).flatten()
moderate_df = pd.DataFrame({'min':moderate_data_min, 'max':moderate_data_max})
moderate_df = moderate_df[moderate_df["min"] > 5] #Filtering unreasonably low speeds

# WCSS and Elbow Method
wcss_plot(slow_df, "Moderate Speed Elbow Graph")

# Clustering with optimal clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 15)
kmeans.fit(moderate_df)
y_km = kmeans.fit_predict(moderate_df)

plt.figure()
plt.scatter(moderate_df.loc[:,"min"], moderate_df.loc[:,"max"], c=y_km, alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k')
plt.xlabel("Min Speed (in km/h)")
plt.ylabel("Max Speed (in km/h)")
plt.title("Perceived Moderate Speed")


## Fast
# Obtaining data points 
fast_data = df.iloc[:, fast_col].to_numpy()
fast_data_min = np.array([int(s.split("-")[0]) for s in fast_data]).flatten()
fast_data_max = np.array([int(s.split("-")[1]) for s in fast_data]).flatten()
fast_df = pd.DataFrame({'min':fast_data_min, 'max':fast_data_max})
fast_df = fast_df[fast_df["max"] < 400]  #Filtering unreasonably high speeds(can discuss)

# WCSS and Elbow Method
wcss_plot(slow_df, "Fast Speed Elbow Graph")

# Clustering with optimal clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 15)
kmeans.fit(fast_df)
y_km = kmeans.fit_predict(fast_df)

plt.figure()
plt.scatter(fast_df.loc[:,"min"], fast_df.loc[:,"max"], c=y_km, alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='k')
plt.xlabel("Min Speed (in km/h)")
plt.ylabel("Max Speed (in km/h)")
plt.title("Perceived Fast Speed")
