# check the versions of key python libraries

import numpy as np
import matplotlib
import pandas as pd
import statsmodels as stats

# scikit-learn
import sklearn
import matplotlib.pyplot as plt
from pandas import concat
from util import file_orig,clean_csv_file,file_cleaned


#Clean File Data
#df_clean_orig = clean_csv_file(file_orig)
#print(df_clean_orig.head())

# Descriptive statistics for each column
#df_clean_orig.describe()

#df_clean_orig.to_csv(file_cleaned, index=False)

#Clean File Data
#df_clean_orig = clean_csv_file(file_orig)
#print(df_clean_orig.head())

# Descriptive statistics for each column
#df_clean_orig.describe()

#df_clean_orig.to_csv(file_cleaned, index=False)

# Set up the plotting layout
#fig, ((ax1)) = plt.subplots(nrows=1, ncols=1, figsize = (10,10))
#fig.autofmt_xdate(rotation = 45)

# Actual max temperature measurement
#ax1.plot(df_clean_orig['DATE'], df_clean_orig['AVG_DAILY_TEMP_ALL_HOURS__F'])
#ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('AVG Temp')

# # Temperature from 1 day ago
# ax2.plot(dates, features['temp_1'])
# ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
#
# # Temperature from 2 days ago
# ax3.plot(dates, features['temp_2'])
# ax3.set_xlabel('Date'); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
#
# # Friend Estimate
# ax4.plot(dates, features['friend'])
# ax4.set_xlabel('Date'); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')

plt.tight_layout(pad=2)

#developer utility files
from featureEng import calcPlotFeatureImportance
from util import clean_csv_file,log, select_n_features, derive_nth_day_feature,derive_nth_day_plus1_feature


#import utility file names
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_train,file_test,file_station,file_val_split,createTestAndTrainingSet

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



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


#Add Date and Station Codes back into the dataset for model
features = []
features.append("DATE")
features.append("STATION_CODES")

for feature in top_features:
    features.append(feature)

#create data frame with filtered feature in trained and test data
df = pd.DataFrame(train_data, columns=features)
df["DATE"] = pd.to_datetime(df["DATE"])
# wont work with multiple index with the same date
# df.set_index('DATE')

df_test = pd.DataFrame(test_data, columns=features)
df_test["DATE"] = pd.to_datetime(df_test["DATE"])
# wont work with multiple index with the same date
# df.set_index('DATE')

#create sliding window for next  days  temp
feature_df = df.AVG_DAILY_TEMP_ALL_HOURS__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat = concat([means, feature_df], axis=1)
df_concat.columns = ['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1', 'AVG_DAILY_TEMP_ALL_HOURS__F']
print(df_concat.head(5))

#create sliding window predictors N - 3 days

predictors = [];

def addPredictor(featureName,n):
    name = featureName + "_" + str(n)
    predictors.append(name)
#
# for feature in features:
#     #if feature != 'DATE':
#         if feature == 'AVG_DAILY_TEMP_ALL_HOURS__F':
#             for N in range(1, 4):
#                 derive_nth_day_feature(df, feature, N)
#                 addPredictor(feature, N)

df['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'] = means
new_df = df.copy(deep=True)
new_df_dates = new_df['DATE']
new_df = new_df.drop('DATE', axis=1)
new_df = new_df.dropna(axis=0, how='any')

#******  Test Set *********
feature_df = df_test.AVG_DAILY_TEMP_ALL_HOURS__F
shifted = feature_df.shift(1)
window = shifted.rolling(window=3)
means = window.mean()
df_concat2 = concat([means, feature_df], axis=1)
df_concat2.columns = ['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1', 'AVG_DAILY_TEMP_ALL_HOURS__F']
print(df_concat2.head(5))


# for feature in features:
#     # if feature != 'DATE':
#         if feature == 'AVG_DAILY_TEMP_ALL_HOURS__F':
#             for N in range(1, 4):
#                 derive_nth_day_feature(df_test, feature, N)


#new_df_test = concat([df_test,df_concat2['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1']], axis=1)
df_test['AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1'] = means
new_df_test = df_test.copy(deep=True)
new_df_test_dates = new_df_test['DATE']
new_df_test = new_df_test.drop('DATE', axis=1)
new_df_test = new_df_test.dropna(axis=0, how='any')

print(new_df.shape)
print(new_df_test.shape)

print(new_df.head())
print(new_df_test.head())



# # Labels are the values we want to predict
labels = np.array(new_df['AVG_DAILY_TEMP_ALL_HOURS__F'])
# Remove the labels from the features
# axis 1 refers to the columns
features= np.array(new_df.drop('AVG_DAILY_TEMP_ALL_HOURS__F', axis = 1))
# Saving feature names for later use
feature_list = list(new_df.columns)
feature_list.remove('AVG_DAILY_TEMP_ALL_HOURS__F')
# Convert to numpy array
features = np.array(features)

#*******test baseline features ****
# Labels are the values we want to predict
test_labels = np.array(new_df_test['AVG_DAILY_TEMP_ALL_HOURS__F'])
# Remove the labels from the features
# axis 1 refers to the columns
test_df_base= np.array(new_df_test.drop('AVG_DAILY_TEMP_ALL_HOURS__F', axis = 1))
# Saving feature names for later use
test_feature_list = list(new_df_test.columns)
test_feature_list.remove('AVG_DAILY_TEMP_ALL_HOURS__F')
# Convert to numpy array
test_features = np.array(test_df_base)


# The baseline predictions are the historical averages
baseline_preds = test_features[:, test_feature_list.index('AVG_DAILY_TEMP_ALL_HOURS__F_PLUS_1')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))



X = np.array(new_df.drop('AVG_DAILY_TEMP_ALL_HOURS__F', axis=1))
y = np.array(new_df['AVG_DAILY_TEMP_ALL_HOURS__F']).astype(float)

X_test = np.array(new_df_test.drop('AVG_DAILY_TEMP_ALL_HOURS__F', axis=1))
y_test = np.array(new_df_test['AVG_DAILY_TEMP_ALL_HOURS__F']).astype(float)

print(X.shape)
print(y.shape)

print(X_test.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=100, max_depth=None,
                                random_state=2)

model = rfr.fit(X,y)
prediction = rfr.predict(X_test)

# Calculate the absolute errors
errors = abs(prediction - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



# resuls
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
df_results

# # Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# # Pull out one tree from the forest
# tree = rfr.estimators_[5]
# # Export the image to a dot file
# export_graphviz(tree, out_file = 'tree.dot', feature_names = test_feature_list, rounded = True, precision = 1)
# # Use dot file to create a graph
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# # Write graph to a png file
# graph.write_png('tree.png');

# Limit depth of tree to 2 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3, random_state=42)
rf_small.fit(features, labels)

# Extract the small tree
tree_small = rf_small.estimators_[5]

# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png');


new_df_test['DATE'] = new_df_test_dates
new_df['DATE'] = new_df_dates

# Plot the values
plt.plot(new_df_test['DATE'], new_df_test['AVG_DAILY_TEMP_ALL_HOURS__F'], 'bo', label = 'Actual',ms=5, color='orange')
# Plot the predicted values
plt.plot(new_df_test['DATE'], prediction, 'ro', label = 'Predicted',  ms=5, color='cyan')
plt.xticks(rotation = '60');
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('AVG DAILY TEMP ALL HOURS (F)'); plt.title('Actual and Predicted Values');
plt.show()

