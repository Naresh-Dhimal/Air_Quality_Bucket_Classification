# Air Quality Bucket Prediction
![Air Quality Prediction](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSeQ09ENPkwhTsrbTnTyrIK0Wn3Ph4Rn9C_gooJ4gVLua7-T7PiGVf_mshg3ICIocHyN4&usqp=CAU)

Utilizing an AQI dataset from Kaggle, I applied a ***decision tree classifier*** and conducted some exploratory data analysis to categorize the air quality into one of the six buckets:
 1. Good
 2. Moderate
 3. Satisfactory
 4. Poor
 5. Very Poor
 6. Severe

# Dataset Reference: [Kaggle](https://www.kaggle.com/datasets/amandeepvasistha/air-quality-data)
# About Dataset:
<b>Context:</b>
Air Quality plays a significant factor in maintaining the health of an individual. Hence, monitoring the Air Quality by measuring and documenting the concentration levels of different pollutants is important.

<b>Source:</b>
The dataset have been derived from Central Pollution Control Board of India: : https://cpcb.nic.in/

<b>Inspiration:</b>
This dataset aims to document the pollutant concentration levels in different cities of India at different dates and time during the period of 2015 - 2020. The pollutant concentration levels can be utilized to determine Air Quality Index and conclude on the air quality of India throughout the period. The dataset is aimed to be updated annually with up-to-date values and credible information.

Libraries used:
```
1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib
5. sklearn
6. ipywidgets
```
# Data Profiling and Inspection
1. Overall Information of dataset.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/af9184eb-a18a-47f0-9244-5a1af53e19c1)

2. Missing Values Analysis.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/fdebc015-319a-4b39-87db-8b6350e6079e)

3. Statistical Overview.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/12a9fd5a-0079-426f-815c-b006b6b307e4)

4. Correlation Analysis.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/ffc32ac1-e342-4bab-9800-89f3a218b678)

# Some Exploratory Data Analysis:

1. AQI_Bucket chart.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/0c55f302-8c52-4287-a06d-fab558317ed7)

2. Distribution of PM2.5 and PM10 with respect to AQI_Bucket.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/1ec1d5c6-1c8d-4c0a-a985-ce86251c1532)

3. Distribution of NO2 and NO with respect to AQI_Bucket.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/2749b5e2-7b2c-466a-89a3-1faa79546294)

4. Distribution of O3 with AQI_Bucket.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/00fb93d1-d648-468d-8fce-ef8027956faf)

5. Yearly Analysis.<br>
Credits:
[PARUL PANDEY](https://www.kaggle.com/parulpandey/breathe-india-covid-19-effect-on-pollution)

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/49a50df1-d52c-492e-892a-3c3beaf42fc8)
![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/775582dd-5653-4b20-bbb1-820f118af5ac)
![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/b0527b45-db60-4b54-b44f-37137fa9ca43)

# Data Preparation and Feature Engineering
1. Handling missing values.<br>
 a. handling missing values of PM2.5
```
def fill_pm_2_5(df):
    df4 = df.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for city_item in df4["City"].unique():
        for year_item in df4["Year"].unique():
            for month_item in df4["Month"].unique():
                slice_df = df4[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item)]
                if not slice_df["PM2.5"].isnull().all():  # Check if all values in the slice are NaN
                    median = slice_df["PM2.5"].median()
                    df4.loc[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item), "PM2.5"] = df4.loc[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item), "PM2.5"].fillna(median)
    return df4

# Call the function passing your DataFrame 'df' as an argument
df_filled = fill_pm_2_5(df)

```
 b. handling missing values of PM10
 ```
def fill_pm_2_5(df):
    df4 = df_filled.copy()  # Create a copy of the DataFrame to avoid modifying the original
    for city_item in df4["City"].unique():
        for year_item in df4["Year"].unique():
            for month_item in df4["Month"].unique():
                slice_df = df4[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item)]
                if not slice_df["PM10"].isnull().all():  # Check if all values in the slice are NaN
                    median = slice_df["PM10"].median()
                    df4.loc[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item), "PM10"] = df4.loc[(df4["City"]==city_item) & (df4["Year"]==year_item) & (df4["Month"]==month_item), "PM10"].fillna(median)
    return df4

# Call the function passing your DataFrame 'df' as an argument
df_filled = fill_pm_2_5(df)

```

2. Droping those rows which has all values NULL.
```
columns_to_check = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI", "AQI_Bucket"]

# Drop rows where all of the specified columns contain missing values
df_dropped = df_filled.drop(df_filled[df_filled[columns_to_check].isna().all(axis=1)].index)

```

3. Replace data.
```
# Replaceing remaining missing values by median

clean_dataset['PM2.5']=clean_dataset['PM2.5'].fillna((clean_dataset['PM2.5'].median()))
clean_dataset['PM10']=clean_dataset['PM10'].fillna((clean_dataset['PM10'].median()))
clean_dataset['NO']=clean_dataset['NO'].fillna((clean_dataset['NO'].median()))
clean_dataset['NO2']=clean_dataset['NO2'].fillna((clean_dataset['NO2'].median()))
clean_dataset['NOx']=clean_dataset['NOx'].fillna((clean_dataset['NOx'].median()))
clean_dataset['NH3']=clean_dataset['NH3'].fillna((clean_dataset['NH3'].median()))
clean_dataset['CO']=clean_dataset['CO'].fillna((clean_dataset['CO'].median()))
clean_dataset['SO2']=clean_dataset['SO2'].fillna((clean_dataset['SO2'].median()))
clean_dataset['O3']=clean_dataset['O3'].fillna((clean_dataset['O3'].median()))
clean_dataset['Benzene']=clean_dataset['Benzene'].fillna((clean_dataset['Benzene'].median()))
clean_dataset['Toluene']=clean_dataset['Toluene'].fillna((clean_dataset['Toluene'].median()))
clean_dataset['Xylene']=clean_dataset['Xylene'].fillna((clean_dataset['Xylene'].median()))
```

4. Handling Categorical data.
```
city = pd.get_dummies(clean_dataset["City"], dtype=float)
```
5. Outlier Detection.<br>
a. Outlier visualization of PM10 using boxplot.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/939ea9b4-c987-4ddb-bfaa-aa48539e6ae0)

b. Removing Outlier.
```
def get_iqr(features):
    # calculating q1
    q1 = np.percentile(features, 25)

    # calculating q3
    q3 = np.percentile(features, 75 )

    # caluculating IQR value
    iqr_value = q3 - q1

    # calculating the lower and upper fence
    lower_fence = q1 - 1.5 * iqr_value
    upper_fence = q3 + 1.5 * iqr_value

    # returning the lower and upper fence
    return lower_fence, upper_fence
def replace_outliers(data, features):
    
    # Iterate over each feature
    for feature in features:
        # Calculate the lower and upper bounds of the interquartile range (IQR)
        lower, upper = get_iqr(data[feature])
        # Define the condition for outliers
        outlier_condition = (data[feature] < lower) | (data[feature] > upper)
        # Replace outliers with the median value of the feature
        data.loc[outlier_condition, feature] = data[feature].median()
    
    return data
features =['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
       'Benzene', 'Toluene', 'Xylene', 'AQI']
data_without_outlier =  replace_outliers(df, features)
```
# Classification model used: Logistic Regression, Decision tree
1. Features Selection for classification.
```
X = train_dataset
y = train_data["AQI_Bucket"]
```
2. Train Test Split.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
3. Modeling.<br>
a. Logistic Regression. <br>
Using logistic regression for classification, classification report for validation data:

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/349f1ddd-0257-4ce7-862f-01a8e79defb6)

b. Decision Tree.<br>
Using decision tree for classification, classification report for validation data:

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/333f6bbf-117e-4ad2-91ca-50d9de4cbecf)


### Hyperparameter tunnig take huge time to execute with this dateset of 650000 enteris. Between Logistic Regression and Decision Tree, decision tree stats seems to be better so i prefer Decision tree to create model to deploy for now.







