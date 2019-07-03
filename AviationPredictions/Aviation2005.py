# check the versions of key python libraries
# scipy
import scipy
from jedi.refactoring import inline

import numpy as np
import matplotlib
import pandas as pd
import statsmodels as stats
# scikit-learn
import sklearn
import matplotlib.pyplot as plt

from featureEng import calcPlotFeatureImportance
#import utility functions
from util import clean_csv_file,log, select_n_features, derive_nth_day_feature

#import utility file names
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_train,file_test,file_station,file_val_split,createTestAndTrainingSet

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# load dataset using read_csv()
#not sure if i need squeezed
from pandas import read_csv

#Clean File Data
df_clean_orig = clean_csv_file(file_orig)
print(df_clean_orig.head())

# Descriptive statistics for each column
df_clean_orig.describe()

df_clean_orig.to_csv(file_cleaned, index=False)

#createTestAndTrainingSet
# create train test sets
train = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
train["DATE"] = pd.to_datetime(train["DATE"])
train = train[(train['DATE'] < '2004-01-01')]
train.to_csv(file_train_split,index=False)

test = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
test["DATE"] = pd.to_datetime(test["DATE"])
test = test[test["DATE"].isin(pd.date_range('2004-01-01', '2005-12-31'))]
test.to_csv(file_test_split,index=False)

val = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)
val["DATE"] = pd.to_datetime(val["DATE"])
val = val[(val['DATE'] > '2006-01-01')]
val.to_csv(file_val_split,index=False)

print('Train Dataset:', train.shape)
print('Test Dataset:', test.shape)
print('Val Dataset:', val.shape)

#avg_daily_temp = df_clean_orig['AVG_DAILY_TEMP_ALL_HOURS__F']
#avg_daily_temp.plot(style='o')



# read split train and test dataset into a csv file
train_data = pd.read_csv(file_train_split, sep=',', error_bad_lines=False)
test_data = pd.read_csv(file_test_split, sep=',', error_bad_lines=False)

# Find important features
# used all data to find important features
feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)

feature_data_noDate = feature_data.drop(['DATE'], axis=1)

# Find, Plot and return top features (create final top features csv file)
top_features= calcPlotFeatureImportance("AVG_DAILY_TEMP_ALL_HOURS__F", feature_data_noDate)

#remove because of non linear relationship
# top_features.remove('DEPART_NORMAL__HDDS_FROM_XN__BASE_OF_65_F')
# top_features.remove('DEPART_NORMAL__EDDS_FROM_XN__BASE_OF_65_F')
# top_features.remove('DEPART_NORMAL__CDDS_ALL_HOURS__BASE_OF_65_F')
# top_features.remove('DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F')


# #Add Date and Station Codes back into the dataset for model
# features = []
# features.append("DATE")
# features.append("STATION_CODES")
#
# for feature in top_features:
#     features.append(feature)
#
# #create data frame with filtered feature in trained and test data
# df = pd.DataFrame(train_data, columns=features)
# df["DATE"] = pd.to_datetime(df["DATE"])
# # wont work with multiple index with the same date
# # df.set_index('DATE')
#
# df_test = pd.DataFrame(test_data, columns=features)
# df_test["DATE"] = pd.to_datetime(df_test["DATE"])
#
# #create sliding window predictors N - 3 days
#
# predictors = [];
#
# def addPredictor(featureName,n):
#     name = featureName + "_" + str(n)
#     predictors.append(name)
#
#
# for feature in features:
#     if feature != 'DATE':
#         if feature != 'STATION_CODES':
#             for N in range(1, 4):
#                 derive_nth_day_feature(df, feature, N)
#                 addPredictor(feature, N)
#
# # ToDo: Look into create one method for this
# for feature in features:
#     if feature != 'DATE':
#         if feature != 'STATION_CODES':
#             for N in range(1, 4):
#                 derive_nth_day_feature(df_test, feature, N)
#
# print(predictors)
#
# # ToDo: Do I need to remove date to improve model
# #save new features to data train and test dataframes
# df2 = df[['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES'] + predictors]
# df3 = df_test[['AVG_DAILY_TEMP_ALL_HOURS__F','STATION_CODES'] + predictors]
#
#
#
# #save training and testing data to file for later use
# df2.to_csv(file_train, index=True)
# df3.to_csv(file_test, index=True)
#
#
# # manually set the parameters of the figure to and appropriate size
# plt.rcParams['figure.figsize'] = [16, 21]
#
# # call subplots specifying the grid structure we desire and that
# # the y axes should be shared
# fig, axes = plt.subplots(nrows=10, ncols=3, sharey=True)
#
# # Since it would be nice to loop through the features in to build this plot
# # let us rearrange our data into a 2D array of 6 rows and 3 columns
# arr = np.array(predictors).reshape(10, 3) #was 10,3
#
# # use enumerate to loop over the arr 2D array of rows and columns
# # and create scatter plots of each meantempm vs each feature
# for row, col_arr in enumerate(arr):
#     for col, feature in enumerate(col_arr):
#         axes[row, col].scatter(df2[feature], df2['AVG_DAILY_TEMP_ALL_HOURS__F'])
#         if col == 0:
#             axes[row, col].set(xlabel=feature, ylabel='AVG_DAILY_TEMP_ALL_HOURS__F')
#         else:
#             axes[row, col].set(xlabel=feature)
# plt.show()
#
#
# df2= df2.dropna(axis=0, how='any')
# df3= df3.dropna(axis=0, how='any')
#
#
# import statsmodels.api as sm
# # separate our my predictor variables (X) from my outcome variable y
# X = df2[predictors]
# y = df2['AVG_DAILY_TEMP_ALL_HOURS__F']
#
# X_test = df3[predictors]
# y_test = df3['AVG_DAILY_TEMP_ALL_HOURS__F']
#
# # # Add a constant to the predictor variable set to represent the Bo intercept
# # X = sm.add_constant(X)
# # X.iloc[:5, :5]
# #
# # X_test = sm.add_constant(X_test)
# # X_test. iloc[:5, :5]
# #
# # # (1) select a significance value
# # alpha = 0.05
# #
# # # (2) Fit the model
# # model = sm.OLS(y, X).fit()
# #
# # # (3) evaluate the coefficients' p-values
# # print(model.summary())
#
#
#
# # (3) cont. - Identify the predictor with the greatest p-value and assess if its > our selected alpha.
# #             based off the table it is clear that meandewptm_3 has the greatest p-value and that it is
# #             greater than our alpha of 0.05
#
# # (4) - Use pandas drop function to remove this column from X
# #X_test = X_test.drop(['NORMAL_AVG_DAILY_WET_BULB_TEMP_ALL_HOURS__F_2','NORMAL_GROWING_DEGREE_DAYS_XN__BASE_OF_50_F_2','NORMAL_GROWING_DEGREE_DAYS_XN__BASE_OF_50_F_3'], axis=1)
# #X = X.drop(['DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_2','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1'],axis=1)
# # X = X.drop(['DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_2','DEPART_NORMAL__EDDS_ALL_HOURS__BASE_OF_65_F_1'],axis=1)
#
# # (5) Fit the model
# model = sm.OLS(y, X).fit()
#
# print(model.summary())
#
#
# from sklearn.model_selection import train_test_split
# # first remove the const column because unlike statsmodels, SciKit-Learn will add that in for us
# #X = X.drop('const', axis=1)
#
# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor(n_estimators=100, max_depth=None,
#                                 random_state=2)
#
# model = rfr.fit(X,y)
# prediction = rfr.predict(X_test)
#
# #plt.plot(rfr.predict(X_test), 'b^', label='RandomForestRegressor')
#
# from sklearn.metrics import mean_absolute_error, median_absolute_error
# print("The Explained Variance: %.2f" % rfr.score(X_test, y_test))
# print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
# print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
#
#
#
# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
#
# # from sklearn.linear_model import LinearRegression
# # # instantiate the regressor class
# # regressor = LinearRegression()
# #
# # # fit the build the model by fitting the regressor to the training data
# # regressor.fit(X, y)
# #
# # # make a prediction set using the test set
# # prediction = regressor.predict(X_test)
# #
# # # Evaluate the prediction accuracy of the model
# # from sklearn.metrics import mean_absolute_error, median_absolute_error
# # print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
# # print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
# # print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
# #
# #
# # df_results = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
# # df_results
#
# # plt.scatter(X_test, y_test,  color='gray')
# # plt.plot(X_test, y_pred, color='red', linewidth=2)
# # plt.show()
#
# # plt.figure()
# # plt.plot(reg1.predict(xt), 'gd', label='GradientBoostingRegressor')
# # plt.plot(reg2.predict(xt), 'b^', label='RandomForestRegressor')
# # plt.plot(reg3.predict(xt), 'ys', label='LinearRegression')
# # plt.plot(ereg.predict(xt), 'r*', label='VotingRegressor')
# # plt.tick_params(axis='x', which='both', bottom=False, top=False,
# #                 labelbottom=False)
# # plt.ylabel('predicted')
# # plt.xlabel('training samples')
# # plt.legend(loc="best")
# # plt.title('Comparison of individual predictions with averaged')
# # plt.show()