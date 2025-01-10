# Predicting Flight Delays and Cancellations: An Integrated Analysis of Airport Data and Weather Data

## Web-App

The web-app link to interact with the spotify recommendation system can be found here: https://airport-weather-prediction.onrender.com

*Note: This service is currently suspended.*

Here are some videos to showcase the web-app.

**Weather Data Exploration**

https://github.com/user-attachments/assets/5ac76d71-cec0-4f39-84ec-0a4b524ca9ee

**Airport Data Exploration**

https://github.com/user-attachments/assets/b566e9a1-032e-4976-b7cb-9813e7c3bb08

**Delay and Cancellation Prediction**

https://github.com/user-attachments/assets/2ea7fb16-7afd-4122-9443-b11481268e45

## Introduction

The holiday season (November to January) is among the busiest for air travel. This project identifies vital flight delays and cancellations patterns, providing actionable insights to help travelers navigate potential disruptions. This report outlines the data preparation steps, including cleaning, merging, and imputation, as well as practical booking tips and predictive models for forecasting delays and cancellations.

## Automated Data Scraping and Engineering

**TRANSTAT BTS Airport Data**

The script [scrape_raw_airport_data.py](https://github.com/Stochastic1017/Airport-Weather-Prediction/blob/e1dc2db7893b3359aed84fd58ac42fa320c3a421/scraping/transtat-bts/scrape_raw_airport_data.py) automates the download of *23 GB* of raw airport data from January 2018 to August 2024. Essential cleaning steps include: 

* Retaining data from January to December. 
* Removing duplicate *AIRPORT_ID* and saving each airport's data separately. 
* Standardizing times to UTC.
* Removing redundant entries for unused *AIRPORT_ID*.

**NCEI LCD Weather Data**

This [scrape_climatology_access.py](https://github.com/Stochastic1017/Airport-Weather-Prediction/blob/e1dc2db7893b3359aed84fd58ac42fa320c3a421/scraping/ncei-lcd/scrape_climatology_access.py) automates the download of \texttt{30 GB} weather data from 2018-2024. Metadata for each station is fetched using [reverse geocoding](https://github.com/thampiman/reverse-geocoder). Key cleaning steps include: 

* Retaining only data from January 2018 to December 2024.
* Removing non-U.S. stations.
* Standardizing times to UTC.

**Merging and Imputation**

With a total 385 airports and 2,580 weather stations, this [script](https://github.com/Stochastic1017/Airport-Weather-Prediction/tree/7ba1676bfc7d28b9bcac61f89f612cb89db3dbbb/miscellaneous_py/merged) uses Haversine distances to identify the 10 closest weather stations per airport. Data is merged using this [script](https://github.com/Stochastic1017/Airport-Weather-Prediction/blob/a589f32dd1da1e8a67876d28a0cbabe29030cedc/dataset/merged/merge_airport_weather.py) as follows: 

* Daily averages of weather features for the closest (within 100 km) weather stations are computed, with sky conditions imputed using the mode.
* Computed delay metrics (arrival, departure, taxi, and total delay) and column merged with nearest weather data within a 30-minute interval prior to estimated departure.
* Any remaining empty weather data is interpolated linearly, and relevant feature engineering is performed to make the data `scikit-learn` trainable.

## Setting up Back-End

In order to allow for data pull, a back-end was set up google cloud bucket. Furthermore, the dataset is also available publicly on kaggle for people to fit prediction models and conduct rigorous statistical analysis.

Airport data: https://www.kaggle.com/datasets/shrivatssudhir/airline-data-bureau-of-transportation-statistics
Weather data: https://www.kaggle.com/datasets/shrivatssudhir/ncei-local-climatology-data-v1
Merged airport and weather data: https://www.kaggle.com/datasets/shrivatssudhir/airport-weather-prediction

## Regression and Classification models

To predict delay using regression, and cancellation using classification, we use Random Forests Regressor and Random Forest Classifier respectively. The code to fit these models can be found [here](https://github.com/Stochastic1017/Airport-Weather-Prediction/blob/main/models/optimized-flight-prediction-v2.py). Both these models are pickled and deployed (in cache) when the web-app is launched.

The RMSE for delay predictions (unbounded) is 156 minutes. The RMSE for delay predictions (bounded), i.e., we only report positive delays is 25 minutes. The cancellation classification has an accuracy of 98.04%.
