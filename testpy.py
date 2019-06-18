# from datetime import datetime, timedelta
# import time
# from collections import namedtuple
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
print(__doc__)
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from featureEng import calcPlotFeatureImportance
from util import log, select_n_features, derive_nth_day_feature

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


#CSV FILE Names
file_cleaned = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"
file_top_features = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_topfeatures.csv"

# 2 Find, Plot and return top features (create final top features csv file)
feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False, index_col=False, dtype='unicode' )

#feature_data = feature_data[(feature_data.YEAR == 2006) & (feature_data.STATION_CODES == 0)]
#feature_data = feature_data.drop(['YEAR'], axis=1)

top_features= calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F", feature_data)


features = []
features.append("DAY")
features.append("YEAR")
features.append("STATION_CODES")

for feature in top_features:
    features.append(feature)

df_top_feature = select_n_features(features, feature_data)
df_top_feature.to_csv(file_top_features, index=True)

print(df_top_feature.head())

#Lag Features

# tmp = df_top_feature[['AVG_DAILY_TEMP_ALL_HOURS__F', 'DEPART_NORMAL__HDDS_FROM_XN__BASE_OF_65_F']].head(10)
# tmp
#
# # 1 day prior
# N = 1
#
# # target measurement of mean temperature
# feature = 'AVG_DAILY_TEMP_ALL_HOURS__F'
#
# # total number of rows
# rows = tmp.shape[0]
#
# # # a list representing Nth prior measurements of feature
# # # notice that the front of the list needs to be padded with N
# # # None values to maintain the constistent rows length for each N
# nth_prior_measurements = [None]*N + [tmp[feature][i-N] for i in range(N, rows)]
#
# # # make a new column name of feature_N and add to DataFrame
# col_name = "{}_{}".format(feature, N)
# tmp[col_name] = nth_prior_measurements
# tmp

df = pd.DataFrame(feature_data, columns=features).set_index(['DAY'])


for feature in features:
    if feature != 'DAY' | feature != 'YEAR' | feature != 'STATION_CODES':
        for N in range(1, 4):
            derive_nth_day_feature(df, feature, N)


# create lag features
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
series = read_csv(file_top_features, header=0, index_col=0,
parse_dates=True, squeeze=True)
temps = DataFrame(series.values)
dataframe = concat([temps.shift(3), temps.shift(2), temps.shift(1), temps], axis=1)
dataframe.columns = ['t-2', 't-1', 't', 't+1']
print(dataframe.head(5))