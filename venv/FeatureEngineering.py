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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#READ CSV FILE
p_file = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData.csv"

WeatherData = pd.read_csv(p_file, sep=',', error_bad_lines=False, index_col=False, dtype='unicode' );

# CLEAN DATA

# CLEAN COLUMN NAMES
WeatherData.columns = WeatherData.columns.str.strip().str.upper().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('/', '')
WeatherData.columns = WeatherData.columns.str.replace('°', '_').str.replace('-', '').str.replace(',', '_').str.replace('.', '').str.replace('+', 'EST').str.replace('^', 'EXP')

# FORMAT NUMERIC DATA




print(WeatherData.head(5))
print(WeatherData.describe())
#WeatherData_2005 =  WeatherData.loc['2005-01-01':'2006-01-01']
WeatherData.head().dtypes
WeatherData = WeatherData.fillna(0)
WeatherData = WeatherData.drop(['Station','Date','SUNSET (local PM)','SUNRISE (local AM)','WBI 2-4 LETTER CODE (all hours)','WBI 2-DIGIT PREDOMINATE WEATHER (all hours, 0-65)','PREDOMINATE WEATHER TEXT (all hours)','PREDOMINATE WEATHER TEXT (all hours)'], axis=1)


WeatherData = WeatherData.sample(frac=0.2)

#WeatherData.corr()[['AVG DAILY TEMP. (all hours) °F']].sort_values('AVG DAILY TEMP. (all hours) °F')

print(WeatherData.describe())

#### Use Random Forests for Plot the Importance of Features

X=np.array(WeatherData.drop('AVG DAILY TEMP. (all hours) °F',axis=1))
y=np.array(WeatherData['AVG DAILY TEMP. (all hours) °F'])
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

ls = list(WeatherData.columns.values)

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()

# Plot to show the most important 10 Features
p = importances[indices][:10]
q=indices[:10]
plt.figure()
plt.title("Top 10 Features' Importance")
plt.bar(range(10), p,
    color="b", yerr=std[q], align="center")
plt.xticks(range(10), q)
plt.xlim([-1,10])
plt.show()

