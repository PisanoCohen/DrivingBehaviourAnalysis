import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt


## Importing data 
data = pd.read_csv('questionnaire response.csv')
df = data.dropna()
column_names = list(df.columns.values)


# Obtain column index
slow_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝘀𝗹𝗼𝘄 in an urban city setting?")
moderate_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝗺𝗼𝗱𝗲𝗿𝗮𝘁𝗲 in an urban city setting?")
fast_col = df.columns.get_loc("What speed range (in km/h) do you consider as 𝗳𝗮𝘀𝘁 in an urban city setting?")


### To get min max speed for each perceived speed range
## Slow
slow = df.iloc[:, slow_col].value_counts()
slow_df = pd.DataFrame(slow)
slow_df.rename(columns={"What speed range (in km/h) do you consider as 𝘀𝗹𝗼𝘄 in an urban city setting?":"count"}, inplace=True)
slow_df["minmax"] = slow_df.index

# Obtaining min and max speed for perceived slow speeds
slow_df["min"] = [int(s.split("-")[0]) for s in slow_df["minmax"]] 
slow_df["max"] = [int(s.split("-")[1]) for s in slow_df["minmax"]]


## Moderate
moderate = df.iloc[:, moderate_col].value_counts()
moderate_df = pd.DataFrame(moderate)
moderate_df.rename(columns={"What speed range (in km/h) do you consider as 𝗺𝗼𝗱𝗲𝗿𝗮𝘁𝗲 in an urban city setting?":"count"}, inplace=True)
moderate_df["minmax"] = moderate_df.index

# Obtaining min and max speed for perceived moderate speeds
moderate_df["min"] = [int(s.split("-")[0]) for s in moderate_df["minmax"]]
moderate_df["max"] = [int(s.split("-")[1]) for s in moderate_df["minmax"]]
moderate_df = moderate_df[moderate_df["min"] > 5] #Filtering unreasonably low speeds


## Fast 
fast = df.iloc[:, fast_col].value_counts()
fast_df = pd.DataFrame(fast)
fast_df.rename(columns={"What speed range (in km/h) do you consider as 𝗳𝗮𝘀𝘁 in an urban city setting?":"count"}, inplace=True)
fast_df["minmax"] = fast_df.index

# Obtaining min and max speed for perceived fast speeds
fast_df["min"] = [int(s.split("-")[0]) for s in fast_df["minmax"]]
fast_df["max"] = [int(s.split("-")[1]) for s in fast_df["minmax"]]
fast_df = fast_df[fast_df["max"] < 500]  #Filtering unreasonably high speeds(can discuss)


## Scatterplot Graph
# # Slow
# plt.figure()
# plt.scatter(slow_df["min"], slow_df["max"], slow_df["count"]*1.5)
# plt.show()


# # Moderate
# plt.figure()
# plt.scatter(moderate_df["min"], moderate_df["max"], moderate_df["count"]*1.5)
# plt.show()


# # Fast
# plt.figure()
# plt.scatter(fast_df["min"], fast_df["max"], fast_df["count"]*1.5)
# plt.show()


# All three
plt.figure()
plt.scatter(slow_df["min"], slow_df["max"], slow_df["count"], c = 'r', marker = "o", alpha = 0.7, label = "slow")
plt.scatter(moderate_df["min"], moderate_df["max"], moderate_df["count"], c = 'g', marker = "^", alpha = 0.7, label = "moderate")
plt.scatter(fast_df["min"], fast_df["max"], fast_df["count"], c = 'b', marker = "s", alpha = 0.7, label = "fast")
plt.legend(loc = "upper left")
plt.xlabel("Min Speed (in km/h)")
plt.ylabel("Max Speed (in km/h)")
plt.title("Perceived Speed Range")
plt.show()



## Data Understanding 
def get_min_max(df):
    min_ = min(df["min"])
    max_ = max(df["max"])
    return min_, max_

def get_average(df):
    min_avg = np.average(df["min"], weights = df["count"])
    max_avg = np.average(df["max"], weights = df["count"])
    return min_avg, max_avg


# Slow 
slow_mode = slow_df["count"].idxmax()
slow_min, slow_max = get_min_max(slow_df)
slow_min_avg, slow_max_avg = get_average(slow_df)


# Moderate
moderate_mode = moderate_df["count"].idxmax()
moderate_min, moderate_max = get_min_max(moderate_df)
moderate_min_avg, moderate_max_avg = get_average(moderate_df)


# Fast 
fast_mode = fast_df["count"].idxmax()
fast_min, fast_max = get_min_max(fast_df)
fast_min_avg, fast_max_avg = get_average(fast_df)
