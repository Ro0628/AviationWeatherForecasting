
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import mean_absolute_error, median_absolute_error
import matplotlib.pyplot as plt



from featureEng import calcPlotFeatureImportance

#import utility file names
from util import file_orig,file_cleaned,file_test_split,file_train_split,file_train,file_test,file_val_split
file_dt_results= "/Users/ronaldajohnson/PycharmProjects/AviationWeatherForecasting/data/WeatherData_DTResults.csv"


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

cleaned_df = pd.read_csv(file_train_split)
test_df = pd.read_csv(file_test_split)
val_df =  pd.read_csv(file_val_split)

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
features.append("DATE")
features.append("STATION_CODES")
features.append("WBI_24_LETTER_CODE_ALL_HOURS_CODES")

for feature in top_features:
    features.append(feature)

#create data frame with filtered feature in trained and test data
df = pd.DataFrame(cleaned_df, columns=features)
df_dates = df['DATE']
df= df.drop(['DATE'], axis=1)

df_t = pd.DataFrame(test_df, columns=features)
df_t_dates = df_t['DATE']
df_t= df_t.drop(['DATE'], axis=1)

df_v = pd.DataFrame(val_df, columns=features)
df_v_dates = df_v['DATE']
df_v= df_v.drop(['DATE'], axis=1)

print(df.shape)
print(df_t.shape)
print(df_v.shape)

X_train = df[[col for col in df.columns if col != 'WBI_24_LETTER_CODE_ALL_HOURS_CODES']]
y_train = df['WBI_24_LETTER_CODE_ALL_HOURS_CODES']

X_test = df_t[[col for col in df_t.columns if col != 'WBI_24_LETTER_CODE_ALL_HOURS_CODES']]
y_test = df_t['WBI_24_LETTER_CODE_ALL_HOURS_CODES']

X_val = df_t[[col for col in df_t.columns if col != 'WBI_24_LETTER_CODE_ALL_HOURS_CODES']]
y_val = df_t['WBI_24_LETTER_CODE_ALL_HOURS_CODES']

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
df_results['DATE'] = df_t_dates.astype(str)


# Plot the values
plt.plot(df_results['DATE'], df_results['Actual'], 'bo', label = 'Actual',ms=5)
# Plot the predicted values
plt.plot(df_results['DATE'], df_results['Predicted'] , 'ro', label = 'Predicted',  ms=5)
plt.xticks(rotation = '60');
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('Weather Conditions'); plt.title('Actual and Predicted Values');
plt.show()




df_results=df_results.replace({18: 'Windy',17: 'Thunderstorm',16: 'Partly Cloudy and Snow Showers',15: 'Snow',14: 'Showers',13: 'Rain and Snow Showers Mix',12: 'Partly Cloudy and Rain Showers',11: 'Rain and Snow',10: 'Rain',9: 'Partly Cloudy',8: 'Partly Cloudy',7: 'Overcast',6: 'N/A',5: 'Mostly Cloudy',4: 'Mostly Cloudy',3: 'Hazy',2: 'Fog',1: 'Fair',0: 'Clear'})
df_results['STATION_CODES'] = df_t['STATION_CODES']
df_results.reindex()
df_results.to_csv(file_dt_results,index=True)


def label_delay (row):
   if row['Predicted'] == 'Thunderstorm' :
      return 'Delay'
   if row['Predicted'] == 'Hazy' and row['Predicted'] == 'Hazy' :
       return 'Delay'
   if row['Predicted'] == 'Fog':
       return 'Delay'
   if row['Predicted'] == 'Snow':
       return 'Delay'
   if row['Predicted'] == 'Rain and Snow Showers Mix':
       return 'Delay'
   return 'None'

df_results["Predicted_Delay"] = df_results.apply (lambda row: label_delay(row), axis=1)



y_val_pred = clf.predict(X_val)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_val, y_val_pred))
print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_val, y_val_pred))
print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_val, y_val_pred))
