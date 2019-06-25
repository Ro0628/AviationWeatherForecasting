import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
print(__doc__)
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#global file names
file_orig="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData.csv"
file_cleaned="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"
file_train_split="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TrainingSet_Split.csv"
file_test_split="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TestingSet_Split.csv"
file_train="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TrainingSet.csv"
file_test ="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TestingSet.csv"
file_val_split="/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_ValidationSet_Split.csv"
file_station= "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TestingStation.csv"



def returnDateDayOfYear(dateStr):
    datetime(dateStr).timetuple().tm_yday;

def returnDateYear(dateStr):
    return datetime(dateStr).timetuple().tm_year;

def log(strText, strSubject):
    print("**************************** Start Logging *************************")
    print("")
    print("Subject: %s \nDescription: %s \n" % ( strSubject, strText))
    print("")
    print("**************************** End Logging *************************")
    print("")


def clean_csv_file(p_file):
    log("Read Data", "Reading Data ...")
    weather_data =  pd.read_csv(p_file)
    weather_data.columns = weather_data.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('(','').str.replace(')', '').str.replace('/', '')
    weather_data.columns = weather_data.columns.str.replace('°', '_').str.replace('-', '').str.replace(',','_').str.replace('.', '').str.replace('+', 'EST').str.replace('^', 'EXP')

    # FORMAT NUMERIC DATA

    print(weather_data.head(5))
    print(weather_data.describe())
    # WeatherData_2005 =  weather_data.loc['2005-01-01':'2006-01-01']
    weather_data.head().dtypes


    log("Label Encoding","")

    lb_enc = LabelEncoder()

    weather_data["STATION_CODES"] = lb_enc.fit_transform(weather_data["STATION"])

    log("2nd Encoding", "Encoding after Replacing missing values")

    weather_data["WBI_24_LETTER_CODE_ALL_HOURS"] = weather_data["WBI_24_LETTER_CODE_ALL_HOURS"].str.replace('/', "")
    weather_data["WBI_24_LETTER_CODE_ALL_HOURS"]= weather_data["WBI_24_LETTER_CODE_ALL_HOURS"].fillna('NA')
    print(weather_data.WBI_24_LETTER_CODE_ALL_HOURS.unique())

    lb_enc2 = LabelEncoder()
    weather_data["WBI_24_LETTER_CODE_ALL_HOURS_CODES"] = lb_enc2.fit_transform(
        weather_data["WBI_24_LETTER_CODE_ALL_HOURS"])
    print(weather_data.WBI_24_LETTER_CODE_ALL_HOURS_CODES.unique())

    log("Remove Unwanted Columns", "Remove unused features : STATION, SUNSET_LOCAL_PM,SUNRISE_LOCAL_AM,WBI_2DIGIT_PREDOMINATE_WEATHER_ALL_HOURS__065, PREDOMINATE_WEATHER_TEXT_ALL_HOURS")
    # dates = weather_data["DATE"]
    #
    # weather_data["DAY"] = pd.to_datetime(weather_data["DATE"])
    # day = weather_data["DAY"].dt.dayofyear
    # year = weather_data["DAY"].dt.year
    #
    # weather_data["DAY"] = day
    # weather_data["YEAR"] = year

    weather_data = weather_data.drop(
        ['STATION', 'SUNSET_LOCAL_PM', 'SUNRISE_LOCAL_AM','WBI_2DIGIT_PREDOMINATE_WEATHER_ALL_HOURS__065', 'WBI_24_LETTER_CODE_ALL_HOURS', 'PREDOMINATE_WEATHER_TEXT_ALL_HOURS'], axis=1)

    print("Weather cols: " + weather_data.columns)

    log("Imputing/Replacing", "Find and replace missing or null values")


    print("%%%% print missing or Null in columns %%%%")
    #print((weather_data == 0).sum())
    #print((weather_data.isnull()).sum())
    # comparing values before dropping null column
    print("\nColumn number before dropping Null column\n",
          len(weather_data.dtypes))

    print("%%%% remove columns with more than 80 % NaN value(NULL) %%%%")
    limitPer = len(weather_data) * 1
    weather_data=weather_data.dropna(axis=1, thresh=limitPer)

    print("\nColumn number after 80% dropping Null column\n",
          len(weather_data.dtypes))

    print("%%%% Impute columns with missing value using avg %%%%")

    weather_data.fillna(weather_data.mean())

   # print(np.isnan(weather_data).sum())

    print("%%%% print missing or Null in columns %%%%")

    print("\n Final weather_data shape",
          weather_data.shape)

    return weather_data

def select_n_features(featuresarray, originalDataframe):
    log("Util Method", "Filter features")
    filterdfeatures = originalDataframe[featuresarray]
    print(filterdfeatures.head())
    return filterdfeatures

def derive_nth_day_feature(df, feature, N):
    rows = df.shape[0]
    nth_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    col_name = "{}_{}".format(feature, N)
    df[col_name] = nth_prior_measurements


def createTestAndTrainingSet():
    # create train test partition
    # create train test sets
    train = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
    train["DATE"] = pd.to_datetime(train["DATE"])
    train = train[(train['DATE'] < '2004-01-01')]
    train.to_csv(file_train_split, index=False)

    test = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
    test["DATE"] = pd.to_datetime(test["DATE"])
    test = test[test["DATE"].isin(pd.date_range('2004-01-01', '2005-12-31'))]
    test.to_csv(file_test_split, index=False)

    val = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
    val["DATE"] = pd.to_datetime(val["DATE"])
    val = val[(val['DATE'] > '2006-01-01')]
    val.to_csv(file_val_split, index=False)

    print('Train Dataset:', train.shape)
    print('Test Dataset:', test.shape)
    print('Val Dataset:', val.shape)



















