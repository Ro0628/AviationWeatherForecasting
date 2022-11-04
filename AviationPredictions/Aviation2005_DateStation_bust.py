# check the versions of key python libraries
# scipy
import scipy
from jedi.refactoring import inline

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

import matplotlib.pyplot as plt

from util import clean_csv_file
from featureEng import calcPlotFeatureImportance
from util import log, select_n_features, derive_nth_day_feature

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# load dataset using read_csv()
#not sure if i need squeezed
from pandas import read_csv

#CSV FILE Names
file_orig = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData.csv"
file_cleaned = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_cleaned.csv"
file_train = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TrainingSet_Split.csv"
file_test = "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_TestingSet.csv"


# 1 Clean File Data
# df_clean_orig = clean_csv_file(file_orig)
# print(df_clean_orig.head())
#
# df_clean_orig.to_csv(file_cleaned, index=False)

# # create train test partition
# train = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
# train["DATE"] = pd.to_datetime(train["DATE"])
# train = train[(train['DATE'] < '2005-01-01')]
# train = train.reset_index(drop=True)
# train.to_csv(file_train, index=False)
#
# test = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
# test["DATE"] = pd.to_datetime(test["DATE"])
# test = test[(test['DATE'] >= '2005-01-01')]
# test = test.reset_index(drop=True)
# test.to_csv(file_test, index=False)
#
# print('Train Dataset:', train.shape)
# print('Test Dataset:', test.shape)


feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)

train_data = pd.read_csv(file_train, sep=',', error_bad_lines=False)
test_data = pd.read_csv(file_test, sep=',', error_bad_lines=False)


#feature_data = feature_data[(feature_data.YEAR == 2006) & (feature_data.STATION_CODES == 0)]
feature_data_noDate = feature_data.drop(['DATE'], axis=1)

# 2 Find, Plot and return top features (create final top features csv file)
top_features= calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F", feature_data_noDate)


features = []
features.append("DATE")
features.append("STATION_CODES")

for feature in top_features:
    features.append(feature)

# df_top_feature = select_n_features(features, feature_data)
# df_top_feature.to_csv(file_top_features, index=True)
#
# print(df_top_feature.head())

df = pd.DataFrame(train_data, columns=features)
df["DATE"] = pd.to_datetime(df["DATE"])


# # 1 day prior
# N = 1

predictors = [];

def addPredictor(featureName,n):
    name = featureName + "_" + str(n)
    predictors.append(name)



for feature in features:
    if feature != 'DATE':
        if feature != 'STATIONS_CODES':
            for N in range(1, 4):
                derive_nth_day_feature(df, feature, N)
                if feature != 'STATIONS_CODES':
                    addPredictor(feature, N)

# predictors.remove('STATION_CODES_1')
# predictors.remove('STATION_CODES_2')
# predictors.remove('STATION_CODES_3')

df_test = pd.DataFrame(test_data, columns=features)
df_test["DATE"] = pd.to_datetime(df_test["DATE"])

for feature in features:
    if feature != 'DATE':
        if feature != 'STATIONS_CODES':
            for N in range(1, 4):
                derive_nth_day_feature(df_test, feature, N)
                if feature != 'STATIONS_CODES':
                    addPredictor(feature, N)


print(predictors)

df2 = df[['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES'] + predictors]

df3 = df_test[['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES'] + predictors]




# manually set the parameters of the figure to and appropriate size
plt.rcParams['figure.figsize'] = [16, 22]

# call subplots specifying the grid structure we desire and that
# the y axes should be shared
fig, axes = plt.subplots(nrows=10, ncols=3, sharey=True)

# Since it would be nice to loop through the features in to build this plot
# let us rearrange our data into a 2D array of 6 rows and 3 columns
arr = np.array(predictors).reshape(10, 3)

# use enumerate to loop over the arr 2D array of rows and columns
# and create scatter plots of each meantempm vs each feature
for row, col_arr in enumerate(arr):
    for col, feature in enumerate(col_arr):
        axes[row, col].scatter(df2[feature], df2['AVG_DAILY_TEMP_ALL_HOURS__F'])
        if col == 0:
            axes[row, col].set(xlabel=feature, ylabel='AVG_DAILY_TEMP_ALL_HOURS__F')
        else:
            axes[row, col].set(xlabel=feature)
plt.show()


df2= df2.dropna(axis=0, how='any')
df3= df3.dropna(axis=0, how='any')

import statsmodels.api as sm
# separate our my predictor variables (X) from my outcome variable y
X = df2[predictors]
y = df2['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES']

X_test = df3[predictors]
y_test = df3['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)
X.iloc[:5, :5]

# (1) select a significance value
alpha = 0.05

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()



# (3) cont. - Identify the predictor with the greatest p-value and assess if its > our selected alpha.
#             based off the table it is clear that meandewptm_3 has the greatest p-value and that it is
#             greater than our alpha of 0.05

# (4) - Use pandas drop function to remove this column from X
X = X.drop('COLDEST_WINDCHILL_TEMP__F_1', axis=1)
# X = X.drop(['DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_2','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1'],axis=1)
# X = X.drop(['DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_2','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1'],axis=1)

# (5) Fit the model
model = sm.OLS(y, X).fit()

model.summary()



from sklearn.model_selection import train_test_split
# first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
#X = X.drop('const', axis=1)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

from sklearn.linear_model import LinearRegression
# instantiate the regressor class
regressor = LinearRegression()

# fit the build the model by fitting the regressor to the training data
regressor.fit(X, y)

# make a prediction set using the test set
prediction = regressor.predict(X_test)

# Evaluate the prediction accuracy of the model
from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))