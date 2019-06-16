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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#READ CSV FILE

file_name = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"

feature_data = pd.read_csv(file_name, sep=',', error_bad_lines=False, index_col=False, dtype='unicode' )


feature_data[(feature_data.YEAR == 1998) & (feature_data.STATION_CODES == 0)]

#feature_data = feature_data.drop(['YEAR'], axis=1)



calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F",feature_data)

