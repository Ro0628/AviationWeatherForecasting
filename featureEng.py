import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(__doc__)
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from util import select_n_features,log

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def calcPlotFeatureImportance(predictFeature, df):
    #### Use Random Forests for Plot the Importance of Features

    dataframe = df.sample(frac=0.01)
    #dataframe = df
    cols = df.columns
    topFeatures = []
    max_topFeatures = 10



    X = np.array(dataframe.drop(predictFeature, axis=1))
    y = np.array(dataframe[predictFeature]).astype(int)

    print(X.shape)
    print(y.shape)

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
        print("Column Name %s" % (cols[indices[f]]))
        if(f < max_topFeatures):
            topFeatures.append(cols[indices[f]])

    #topFeatures.append(predictFeature)
    print(topFeatures)


        # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature Importance")
    plt.bar(range(X.shape[1]), importances[indices],
            color="b", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    # Plot to show the most important 10 Features
    p = importances[indices][:max_topFeatures]
    q = indices[:max_topFeatures]
    plt.figure()
    plt.title("Top Features' Importance")
    plt.bar(range(max_topFeatures), p,
            color="b", yerr=std[q], align="center")
    plt.xticks(range(max_topFeatures), q)
    plt.xlim([-1, max_topFeatures])
    plt.show()

    return topFeatures