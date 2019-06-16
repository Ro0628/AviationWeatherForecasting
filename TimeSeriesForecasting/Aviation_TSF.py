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

p_file = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData.csv"
series = read_csv(p_file, index_col=0)
print(type(series))
print(series.head())

clean_data = clean_csv_file(p_file)

print(clean_data.head())
file_name = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"

clean_data.to_csv(file_name, index=False)

feature_data = clean_data.copy(deep=False)

feature_data[(feature_data.YEAR == 1998) & (feature_data.STATION_CODES == 0)]
feature_data = feature_data.drop(['YEAR'], axis=1)

print(feature_data.head())

calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F",feature_data)

