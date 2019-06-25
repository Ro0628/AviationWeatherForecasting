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
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_val_split,file_train,file_test


# Create a random dataset

cleaned_df = pd.read_csv(file_train_split)
test_df = pd.read_csv(file_test_split)
val_df = pd.read_csv(file_val_split)

# used all data to find important features
feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)

feature_data_noDate = feature_data.drop(['DATE'], axis=1)

# Find, Plot and return top features (create final top features csv file)
top_features= calcPlotFeatureImportance("WBI_24_LETTER_CODE_ALL_HOURS_CODES", feature_data_noDate)

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
df=df.drop(['DATE'], axis=1)

df_t = pd.DataFrame(test_df)
df_t= df_t.drop(['DATE'], axis=1)

df_v =  pd.DataFrame(val_df)
df_v= df_v.drop(['DATE'], axis=1)




print(df.shape)
print(df_t.shape)
print(df_v.shape)

X_train = df[[col for col in df.columns if col not in ['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]]
y_train = df[['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]

X_test = df_t[[col for col in df_t.columns if col  not in ['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]]
y_test = df_t[['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]


X_val = df_v[[col for col in df_v.columns if col  not in ['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]]
y_val = df_v[['LOW_TEMP__F','HIGH_TEMP_F','AVG_DAILY_TEMP_ALL_HOURS__F']]

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



# # Plot the results
# plt.figure()
# s = 50
# a = 0.4
# plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
#             c="navy", s=s, marker="s", alpha=a, label="Data")
# plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
#             c="cornflowerblue", s=s, alpha=a,
#             label="Multi RF score=%.2f" % regr_multirf.score(X_test, y_test))
# plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
#             c="c", s=s, marker="^", alpha=a,
#             label="RF score=%.2f" % regr_rf.score(X_test, y_test))
# plt.xlim([-6, 6])
# plt.ylim([-6, 6])
# plt.xlabel("target 1")
# plt.ylabel("target 2")
# plt.title("Comparing random forests and the multi-output meta estimator")
# plt.legend()
# plt.show()

from sklearn.metrics import mean_absolute_error, median_absolute_error
print("The Explained Variance: %.2f" % regr_multirf.score(X_val, y_val))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_val, y_multirf))


df_results = pd.DataFrame({'Actual 0': y_val['LOW_TEMP__F'], 'Predicted 0': y_multirf[:,0],'Actual 1': y_val['HIGH_TEMP_F'], 'Predicted 1': y_multirf[:,1],'Actual 2': y_val['AVG_DAILY_TEMP_ALL_HOURS__F'], 'Predicted 2': y_multirf[:,2]})
df_results

