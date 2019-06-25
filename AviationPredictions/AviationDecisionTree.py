# check the versions of key python libraries
# scipy
import scipy
from jedi.refactoring import inline

import numpy as np
import matplotlib
import pandas as pd
import statsmodels as stats
# scikit-learn
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt
from sklearn import tree


from featureEng import calcPlotFeatureImportance
#import utility functions
from util import clean_csv_file,log, select_n_features, derive_nth_day_feature

#import utility file names
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_train,file_test

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

cleaned_df = pd.read_csv(file_train_split)
test_df = pd.read_csv(file_test_split)
#print(train_df.shape)train_df=train_df['DATE','STATION','AVG_DAILY_TEMP_ALL_HOURS_F','COLDEST_WINDCHILL_TEMP__F','NORMAL_AVG_DAILY_WET_BULB_TEMP_ALL_HOURS__F','HIGH_TEMP_F','DFN_AVG_DAILY_TEMP_XN__F']
# Find important features
# used all data to find important features
feature_data = pd.read_csv(file_cleaned, sep=',', error_bad_lines=False)

feature_data_noDate = feature_data.drop(['DATE'], axis=1)

# Find, Plot and return top features (create final top features csv file)
top_features= calcPlotFeatureImportance("WBI_24_LETTER_CODE_ALL_HOURS_CODES", feature_data_noDate)

cleaned_df.dropna(axis=0,how='any')


#Add Date and Station Codes back into the dataset for model
features = []
#features.append("DATE")
features.append("STATION_CODES")
features.append("WBI_24_LETTER_CODE_ALL_HOURS_CODES")

for feature in top_features:
    features.append(feature)

#create data frame with filtered feature in trained and test data
df = pd.DataFrame(cleaned_df, columns=features)


df_t = pd.DataFrame(test_df, columns=features)


print(df.shape)
print(df_t.shape)

X_train = df[[col for col in df.columns if col != 'WBI_24_LETTER_CODE_ALL_HOURS_CODES']]
y_train = df['WBI_24_LETTER_CODE_ALL_HOURS_CODES']

X_test = df_t[[col for col in df_t.columns if col != 'WBI_24_LETTER_CODE_ALL_HOURS_CODES']]
y_test = df_t['WBI_24_LETTER_CODE_ALL_HOURS_CODES']

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)



from sklearn.multioutput import MultiOutputRegressor
#Predict the response for test dataset
y_pred = clf.predict(X_test)





# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, y_pred))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, y_pred))

df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_results



#tree.plot_tree(clf.fit(X_train,y_train))

# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("Tree")

#
# dot_data = tree.export_graphviz(clf, out_file=None,
# ...                      feature_names=iris.feature_names,
# ...                      class_names=iris.target_names,
# ...                      filled=True, rounded=True,
# ...                      special_characters=True)
# >>> graph = graphviz.Source(dot_data)
# >>> graph