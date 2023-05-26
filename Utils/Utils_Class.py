##=================================IMPORTS=================================##
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBRegressor, XGBClassifier

import sys
import os
import json

from constants import *


import warnings
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

##=================================CLASS=================================##

class Utils:

    def __init__(self, matchList) -> None:
        self.TARGET = 'Score'
        self.CURRENT_YEAR = 2022
        self.CURRENT_SEMESTER = 1
        self.CURRENT_SEMESTER_YEAR = str(CURRENT_YEAR)+str(CURRENT_SEMESTER)
        self.LAST_SEMESTER = abs(CURRENT_SEMESTER-1)
        self.LAST_YEAR = CURRENT_YEAR-1 if LAST_SEMESTER==1 else CURRENT_YEAR