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


# Demographics DF
demographics = df.iloc[:,2:4]
demographics["Gender_Int"] = pd.Categorical(demographics.loc[:,"Gender"])
demographics["Gender_Int"] = demographics["Gender_Int"].cat.codes # Female = 0, Male = 1, No say = 2


# Driving experience DF
driving = df.iloc[:,5:8]
driving["Days"] = pd.Categorical(driving.iloc[:,1])
driving["Days"].replace(["0 to 2 days", "3 to 5 days", "6 to 7 days"], [0, 1, 2], inplace = True)
driving["Mileage"] = pd.Categorical(driving.iloc[:,2])
driving["Mileage"].replace(["<10000 km", "10000- 50000 km", "50000 - 100000 km", ">100000 km"], [0, 1, 2, 3], inplace = True)



### Measure of central tendency/distribution of data
## Measure of central tendency 
# Demographics 
age_mean = np.mean(demographics["Age"])
gender_mean = np.mean(demographics["Gender_Int"])

# Driving Experience 
years_mean = np.mean(driving["Years Driving."])
days_mean = np.mean(driving["Days"])
days_count = driving["Days"].value_counts()
mileage_mean = np.mean(driving["Mileage"])
mileage_count = driving["Mileage"].value_counts()

# Baseline response 
responses_count = responses_df.stack().value_counts()
fast_count = fast_df.stack().value_counts()
moderate_count = moderate_df.stack().value_counts()
slow_count = slow_df.stack().value_counts()


## Plotting Distribution of data
def distribution_plot(df, column, title, xlabel = "Class"):
    plt.figure()
    df[column].groupby(df[column]).hist()
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)

# Demographics 
distribution_plot(demographics, "Age", "Age Data Distribution", "Age")
distribution_plot(driving, "Years Driving.", "Driving Years Data Distribution", "Years")
distribution_plot(driving, "Days", "Driving Days in a Week Data Distribution")
distribution_plot(driving, "Mileage", "Lifetime Mileage Data Distribution")



### Performing Clustering     
# # All scenarios
wcss_plot(responses_df, "All 42 Scenarios")

kmeans = KMeans(n_clusters = 4, random_state = 15)
kmeans.fit(responses_df)
y_responses_km = kmeans.fit_predict(responses_df)


# # Fast Scenarios
wcss_plot(fast_df, "13 Fast Scenarios")

kmeans = KMeans(n_clusters = 4, random_state = 15)
kmeans.fit(fast_df)
y_fast_km = kmeans.fit_predict(fast_df)


# # Moderate Scenarios 
wcss_plot(moderate_df, "13 Moderate Scenarios")

kmeans = KMeans(n_clusters = 4, random_state = 15)
kmeans.fit(moderate_df)
y_moderate_km = kmeans.fit_predict(moderate_df)


## Slow Scenarios
wcss_plot(slow_df, "13 Slow Scenarios")

kmeans = KMeans(n_clusters = 3, random_state = 15)
kmeans.fit(slow_df)
y_slow_km = kmeans.fit_predict(slow_df)



### Linking data to their demographics
## Functions
def get_class_idx(results_km):
    main_arr = []
    for i in range (0, len(np.unique(results_km))):
        arr = []
        for j in range (0, len(results_km)):
            if results_km[j] == i:
                arr.append(j)
        main_arr.append(arr)
    return main_arr

def get_demographics(classes_idx):
    all_classes = []
    for i in range (0, len(classes_idx)):
        demo_age = np.array([demographics.iloc[i,0] for i in classes_idx[i]]).flatten()
        demo_gender = np.array([demographics.iloc[i,2] for i in classes_idx[i]]).flatten()
        class_demo = pd.DataFrame({'Age':demo_age, 'Gender':demo_gender})
        all_classes.append(class_demo)
    return all_classes

def get_driving(classes_idx):
    all_classes = []
    for i in range (0, len(classes_idx)):  
        drive_years = np.array([driving.iloc[i,0] for i in classes_idx[i]]).flatten()
        drive_days = np.array([driving.iloc[i,3] for i in classes_idx[i]]).flatten()
        drive_mileage = np.array([driving.iloc[i,4] for i in classes_idx[i]]).flatten()
        class_driving = pd.DataFrame({'Years':drive_years, 'Days':drive_days, 'Mileage':drive_mileage})
        all_classes.append(class_driving)
    return all_classes


# All scenarios 
responses_idx       = get_class_idx(y_responses_km)
responses_demo      = get_demographics(responses_idx)
responses_driving   = get_driving(responses_idx)

# Fast scenarios 
fast_class_idx      = get_class_idx(y_fast_km)
fast_class_demo     = get_demographics(fast_class_idx)
fast_class_driving  = get_driving(fast_class_idx)

# Moderate scenarios 
moderate_class_idx  = get_class_idx(y_moderate_km)
moderate_class_demo = get_demographics(moderate_class_idx)
moderate_class_driving = get_driving(moderate_class_idx)

# Slow scenarios 
slow_class_idx      = get_class_idx(y_slow_km)
slow_class_demo     = get_demographics(slow_class_idx)
slow_class_driving  = get_driving(slow_class_idx)



### Understanding the clusters
## Functions
def age_understanding(demo_df):
    main_arr = []
    for i in range (0, len(demo_df)):
        temp_arr = []
        temp_arr.append(min(demo_df[i]["Age"]))
        temp_arr.append(max(demo_df[i]["Age"]))
        temp_arr.append(np.mean(demo_df[i]["Age"]))
        main_arr.append(temp_arr)
    return main_arr

def gender_understanding(demo_df):
    main_arr = []
    for i in range (0, len(demo_df)):
        temp_arr = []
        temp_arr.append(demo_df[i]["Gender"].value_counts())
        temp_arr.append(np.mean(demo_df[i]["Gender"]))
        main_arr.append(temp_arr)
    return main_arr

def years_understanding(driving_df):
    main_arr =[]
    for i in range (0, len(driving_df)):
        temp_arr = []
        temp_arr.append(min(driving_df[i]["Years"]))
        temp_arr.append(max(driving_df[i]["Years"]))
        temp_arr.append(np.mean(driving_df[i]["Years"]))
        main_arr.append(temp_arr)
    return main_arr

def days_understanding(driving_df):
    main_arr =[]
    for i in range (0, len(driving_df)):
        temp_arr = []
        temp_arr.append(np.mean(driving_df[i]["Days"]))
        main_arr.append(temp_arr)
    return main_arr

def mileage_understanding(driving_df):
    main_arr =[]
    for i in range (0, len(driving_df)):
        temp_arr = []
        temp_arr.append(np.mean(driving_df[i]["Mileage"]))
        main_arr.append(temp_arr)
    return main_arr


# All scenarios 
responses_age_u     = age_understanding(responses_demo)
responses_gender_u  = gender_understanding(responses_demo)
responses_years_u   = years_understanding(responses_driving)
responses_days_u    = days_understanding(responses_driving)
responses_mileage_u = mileage_understanding(responses_driving)

# Fast scenarios 
fast_age_u          = age_understanding(fast_class_demo)
fast_gender_u       = gender_understanding(fast_class_demo)
fast_years_u        = years_understanding(fast_class_driving)
fast_days_u         = days_understanding(fast_class_driving)
fast_mileage_u      = mileage_understanding(fast_class_driving)

# Moderate scenarios 
moderate_age_u      = age_understanding(moderate_class_demo)
moderate_gender_u   = gender_understanding(moderate_class_demo)
moderates_years_u   = years_understanding(moderate_class_driving)
moderate_days_u     = days_understanding(moderate_class_driving)
moderate_mileage_u  = mileage_understanding(moderate_class_driving)

# Slow scenarios
slow_age_u          = age_understanding(slow_class_demo)
slow_gender_u       = gender_understanding(slow_class_demo)
slow_years_u        = years_understanding(slow_class_driving)
slow_days_u         = days_understanding(slow_class_driving)
slow_mileage_u      = mileage_understanding(slow_class_driving)
