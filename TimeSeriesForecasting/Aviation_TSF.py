# check the versions of key python libraries
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy as np
print('numpy: %s' % np.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas as pd
print('pandas: %s' % pd.__version__)
# statsmodels
import statsmodels as stats
print('statsmodels: %s' % stats.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)

from util import clean_csv_file
from featureEng import calcPlotFeatureImportance

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# load dataset using read_csv()
#not sure if i need squeezed
from pandas import read_csv

file_orig = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData.csv"
file_cleaned = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"
file_top_features = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_topfeatures.csv"

# 1 Clean File Data
df_clean_orig = clean_csv_file(file_orig)
print(df_clean_orig.head())

df_clean_orig.to_csv(file_cleaned, index=False)

# 2 Find, Plot and return top features
# feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False, index_col=False, dtype='unicode' )
#
# feature_data[(feature_data.YEAR == 1998) & (feature_data.STATION_CODES == 0)]
# feature_data = feature_data.drop(['YEAR'], axis=1)
#
# df_top_feature = calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F", feature_data)



