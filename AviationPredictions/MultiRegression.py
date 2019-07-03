#Ronalda Johnson

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from featureEng import calcPlotFeatureImportance
#import utility file names
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_val_split
from util import file_train,file_test
from pandas import concat

file_mr_results= "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_MultiRegressionResults.csv"
# Create a random dataset

cleaned_df = pd.read_csv(file_train_split)
test_df = pd.read_csv(file_test_split)
val_df = pd.read_csv(file_val_split)

# used all data to find important features
feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)

feature_data_noDate = feature_data.drop(['DATE'], axis=1)

# Find, Plot and return top features (create final top features csv file)
top_features= calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F", feature_data_noDate)

cleaned_df.dropna(axis=0,how='any')


#correlation
cor1=cleaned_df[cleaned_df.columns[1:]].corr()['AVG_DAILY_TEMP_ALL_HOURS__F'][:]
cor2=cleaned_df[cleaned_df.columns[1:]].corr()['HIGH_TEMP_F'][:]
cor3=cleaned_df[cleaned_df.columns[1:]].corr()['LOW_TEMP__F'][:]

cor1Sort1 = cor1.sort_values(ascending=False)
cor1Sort2 = cor2.sort_values(ascending=False)
cor1Sort3 = cor3.sort_values(ascending=False)

#Add Date and Station Codes back into the dataset for model
features = []
#features.append("DATE")
features.append("STATION_CODES")
features.append("WBI_24_LETTER_CODE_ALL_HOURS_CODES")
features.append("HIGH_TEMP_F")
features.append('AVG_DAILY_TEMP_ALL_HOURS__F')

for feature in top_features:
    features.append(feature)

#create data frame with filtered feature in trained and test data
df = pd.DataFrame(cleaned_df)

df_t = pd.DataFrame(test_df)

df_v =  pd.DataFrame(val_df)


#*** Train ***
#create sliding window for next  days  temp
feature_df = df.AVG_DAILY_TEMP_ALL_HOURS__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1', 'AVG_DAILY_TEMP_ALL_HOURS__F']
print(df_concat.head(5))

df['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'] = means


feature_df = df.HIGH_TEMP_F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['HIGH_TEMP_F_PLUS_1', 'HIGH_TEMP_F']
print(df_concat.head(5))

df['HIGH_TEMP_F_PLUS_1'] = means

feature_df = df.LOW_TEMP__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['LOW_TEMP_F_PLUS_1', 'LOW_TEMP_F']
print(df_concat.head(5))

df['LOW_TEMP_F_PLUS_1'] = means
df = df.dropna(axis=0, how='any')

#*** test ****
#create sliding window for next  days  temp
feature_df = df_t.AVG_DAILY_TEMP_ALL_HOURS__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1', 'AVG_DAILY_TEMP_ALL_HOURS__F']
print(df_concat.head(5))

df_t['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'] = means


feature_df = df_t.HIGH_TEMP_F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['HIGH_TEMP_F_PLUS_1', 'HIGH_TEMP_F']
print(df_concat.head(5))

df_t['HIGH_TEMP_F_PLUS_1'] = means

feature_df = df_t.LOW_TEMP__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['LOW_TEMP_F_PLUS_1', 'LOW_TEMP_F']
print(df_concat.head(5))

df_t['LOW_TEMP_F_PLUS_1'] = means
df_t = df_t.dropna(axis=0, how='any')



#*** val ****
#create sliding window for next  days  temp
feature_df = df_v.AVG_DAILY_TEMP_ALL_HOURS__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1', 'AVG_DAILY_TEMP_ALL_HOURS__F']
print(df_concat.head(5))

df_v['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'] = means


feature_df = df_v.HIGH_TEMP_F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['HIGH_TEMP_F_PLUS_1', 'HIGH_TEMP_F']
print(df_concat.head(5))

df_v['HIGH_TEMP_F_PLUS_1'] = means

feature_df = df_v.LOW_TEMP__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['LOW_TEMP_F_PLUS_1', 'LOW_TEMP_F']
print(df_concat.head(5))

df_v['LOW_TEMP_F_PLUS_1'] = means
df_v = df_v.dropna(axis=0, how='any')


print(df.shape)
print(df_t.shape)
print(df_v.shape)

df_dates = df['DATE']
df_stations = df['STATION_CODES']
df=df.drop(['DATE'], axis=1)

df_t_dates = df_t['DATE']
df_t_stations = df_t['STATION_CODES']
df_t= df_t.drop(['DATE'], axis=1)

df_v_dates = df_v['DATE']
df_v_stations = df_v['STATION_CODES']
df_v= df_v.drop(['DATE'], axis=1)


X_train = df[[col for col in df.columns if col not in ['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]]
y_train = df[['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]

X_test = df_t[[col for col in df_t.columns if col  not in  ['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]]
y_test = df_t[['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]


X_val = df_v[[col for col in df_v.columns if col  not in ['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]]
y_val = df_v[['LOW_TEMP_F_PLUS_1','HIGH_TEMP_F_PLUS_1','AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']]

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)



max_depth = None #30
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,
                                                          max_depth=max_depth,
                                                       random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=100, max_depth=max_depth,
                                random_state=2)
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_val)

#y_rf = regr_rf.predict(X_test)



from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regr_multirf.score(X_val, y_val))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_val, y_multirf))


df_results = pd.DataFrame({'Actual Low': y_val['LOW_TEMP_F_PLUS_1'], 'Predicted Low': y_multirf[:,0],'Actual High': y_val['HIGH_TEMP_F_PLUS_1'], 'Predicted High': y_multirf[:,1],'Actual Avg Daily Temp': y_val['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'], 'Predicted Actual Avg Daily Temp': y_multirf[:,2]})
df_results['DATE'] = df_t_dates.astype(str)
df_results['STATION_CODES'] = df_t_stations
df_results.reindex()
df_results.to_csv(file_mr_results,index=True)

print(df_results)


