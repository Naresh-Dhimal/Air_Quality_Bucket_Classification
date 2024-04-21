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

3. Distribution of NO2 and No with respect to AQI_Bucket.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/2749b5e2-7b2c-466a-89a3-1faa79546294)

4. Distribution of O3 with AQI_Bucket.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/00fb93d1-d648-468d-8fce-ef8027956faf)

5. Yearly Analysis.

![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/49a50df1-d52c-492e-892a-3c3beaf42fc8)
![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/775582dd-5653-4b20-bbb1-820f118af5ac)
![image](https://github.com/Naresh-Dhimal/Air_Quality_Bucket_Classification/assets/122601911/b0527b45-db60-4b54-b44f-37137fa9ca43)











