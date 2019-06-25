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
from featureEng import calcPlotFeatureImportance
from util import log, select_n_features, derive_nth_day_feature,createTestAndTrainingSet



createTestAndTrainingSet