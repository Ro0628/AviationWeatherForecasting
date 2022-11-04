#import libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import explained_variance_score,mean_absolute_error, median_absolute_error

#import file names
from file_names import file_cleaned, file_test,file_train



#import data

df_weather_train_set = pd.read_csv(file_train)
df_weather_train_set.describe().T
df_weather_train_set.info()
df_weather_train_set_filtered = df_weather_train_set[(df_weather_train_set['STATION_CODES'] == 3)]
df_weather_train_set_filtered=df_weather_train_set_filtered.drop('Unnamed: 0', axis=1)

# SPLIT TESTING AND VALIDATION SET BY SPLITTING LOCATIONS FOR YEARS AFTER 2005
df_weather_test_set = pd.read_csv(file_test)

# ToDo: Not sure whereS the extra 0 is coming from but must remove it
df_weather_test_set=df_weather_test_set.drop('Unnamed: 0', axis=1)
df_weather_test_set_validation = df_weather_test_set[(df_weather_test_set['STATION_CODES'] == 4) | (df_weather_test_set['STATION_CODES'] == 12)]
df_weather_test_set = df_weather_test_set[(df_weather_test_set['STATION_CODES'] == 11)]

frames = [df_weather_train_set_filtered, df_weather_test_set_validation, df_weather_test_set]
df_weather_all = pd.concat(frames)

print(df_weather_train_set_filtered.shape)
print(df_weather_test_set.shape)
print(df_weather_test_set_validation.shape)
print(df_weather_all.shape)

#combined weather dataset
X = df_weather_all[[col for col in df_weather_all.columns if col != 'AVG_DAILY_TEMP_ALL_HOURS__F']]
y  = df_weather_all['AVG_DAILY_TEMP_ALL_HOURS__F']

# Train, Validate & Test sets

# X will be a pandas dataframe of all columns except AVG_DAILY_TEMP_ALL_HOURS__F
# y will be a pandas series of the AVG_DAILY_TEMP_ALL_HOURS__F
X_val = df_weather_test_set_validation[[col for col in df_weather_test_set_validation.columns if col != 'AVG_DAILY_TEMP_ALL_HOURS__F']]
y_val = df_weather_test_set_validation['AVG_DAILY_TEMP_ALL_HOURS__F']

X_test = df_weather_test_set[[col for col in df_weather_test_set.columns if col != 'AVG_DAILY_TEMP_ALL_HOURS__F']]
y_test = df_weather_test_set['AVG_DAILY_TEMP_ALL_HOURS__F']

X_train= df_weather_train_set_filtered[[col for col in df_weather_train_set_filtered.columns if col != 'AVG_DAILY_TEMP_ALL_HOURS__F']]
y_train = df_weather_train_set_filtered['AVG_DAILY_TEMP_ALL_HOURS__F']

feature_cols = [tf.feature_column.numeric_column(col) for col in X.columns]
print(feature_cols)

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[150, 150],
                                      model_dir='tf_wx_model')

def wx_input_fn(X, y=None, num_epochs=None, shuffle=True, batch_size=400):
    return tf.estimator.inputs.pandas_input_fn(x=X,
                                               y=y,
                                               num_epochs=num_epochs,
                                               shuffle=shuffle,
                                               batch_size=batch_size)

evaluations = []
STEPS = 400
for i in range(100):
    regressor.train(input_fn=wx_input_fn(X_train, y=y_train), steps=STEPS)
    evaluations.append(regressor.evaluate(input_fn=wx_input_fn(X_val,
                                                               y_val,
                                                               num_epochs=1,
                                                               shuffle=False)))

