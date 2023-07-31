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
start_col = df.columns.get_loc("Sunny, minimal traffic. Roughly 1 cars' length to the intersection. [Fast]")
end_col = df.columns.get_loc("Traffic light with countdown timer, 3 seconds for yellow light. Sunny, high traffic. Roughly 1 cars' length to the intersection.  [Slow]")
feature_no = 1

# Changing features (categorical) to int 
while start_col < end_col+1: 
    col_name = "NewCol"
    col_name = col_name + str(feature_no)
    df[col_name] = pd.Categorical(df.iloc[:,start_col])
    df[col_name] = df[col_name].cat.codes # Accelerate = 0, Brake = 1
    feature_no += 1
    start_col += 1


## Obtaining relevant dataframes 
# Driving behaviour response (All scenarios)
responses_start = df.columns.get_loc("NewCol1")
responses_end = df.columns.get_loc("NewCol42")
responses_df = df.iloc[:,responses_start:responses_end+1]


# Speed columns index
fast_loc = [i for i in range (0,42,3)]
moderate_loc = [i for i in range (1,42,3)]
slow_loc = [i for i in range (2,42,3)]

# Respective speed scneario DFs
fast_df = responses_df.iloc[:,fast_loc]
moderate_df = responses_df.iloc[:,moderate_loc]
slow_df = responses_df.iloc[:,slow_loc]



### Performing Clustering     
# All scenarios
wcss_plot(responses_df, "All 42 Scenarios")

responses_kmeans = KMeans(n_clusters = 4, random_state = 15)
responses_kmeans.fit(responses_df)
y_responses_km = responses_kmeans.fit_predict(responses_df)


# Fast Scenarios
wcss_plot(fast_df, "13 Fast Scenarios")

fast_kmeans = KMeans(n_clusters = 4, random_state = 15)
fast_kmeans.fit(fast_df)
y_fast_km = fast_kmeans.fit_predict(fast_df)


# Moderate Scenarios 
wcss_plot(moderate_df, "13 Moderate Scenarios")

moderate_kmeans = KMeans(n_clusters = 4, random_state = 15)
moderate_kmeans.fit(moderate_df)
y_moderate_km = moderate_kmeans.fit_predict(moderate_df)


# Slow Scenarios
wcss_plot(slow_df, "13 Slow Scenarios")

slow_kmeans = KMeans(n_clusters = 3, random_state = 15)
slow_kmeans.fit(slow_df)
y_slow_km = slow_kmeans.fit_predict(slow_df)



### Visualizations of cluster centers
def visualize_cc(kmean, title):
    cc = kmean.cluster_centers_   
    plt.figure()
    for i in range (0, len(cc)):
        plt.plot([i for i in range(0,len(cc[0]))], cc[i], label = f"Cluster {i}")
    plt.legend()
    plt.title(title)
    ax = plt.axes()
    ax.text(-0.03, 0, "Accelerate", transform = ax.transAxes, horizontalalignment = "right")
    ax.text(-0.03, 0.945, "Brake", transform = ax.transAxes, horizontalalignment = "right")
    

visualize_cc(responses_kmeans, "All 42 Scenarios Cluster Center")
visualize_cc(fast_kmeans, "14 Fast Scenarios Cluster Center")
visualize_cc(moderate_kmeans, "14 Moderate Scenarios Cluster Center")
visualize_cc(slow_kmeans, "14 Slow Scenarios Cluster Center")



### Cluster Centers average value
def avg_cc(kmean):
    arr = []
    cc = kmean.cluster_centers_
    for i in range (0, len(cc)):
        arr.append(np.mean(cc[i]))
    return arr

responses_avg_cc = avg_cc(responses_kmeans)
fast_avg_cc = avg_cc(fast_kmeans)
moderate_avg_cc = avg_cc(moderate_kmeans)
slow_avg_cc = avg_cc(slow_kmeans)
