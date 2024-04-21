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
<b>Context</b>
Air Quality plays a significant factor in maintaining the health of an individual. Hence, monitoring the Air Quality by measuring and documenting the concentration levels of different pollutants is important.

<b>Source</b>
The dataset have been derived from Central Pollution Control Board of India: : https://cpcb.nic.in/

<b>Inspiration</b>
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
1. Handling missing values.
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






