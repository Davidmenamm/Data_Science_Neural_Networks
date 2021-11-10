""" Reading and Preprocessing Actions"""

# imports
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter



"""
@param path, location of input dataset file
@param balance, if class balancing is applied to the read dataset
returns the the features and target of the dataset """
def readDataset(inPath, balancing = False):
    # read csv
    df = pd.read_csv(inPath)
    # only features
    X = df.iloc[:, 1: len(df.columns)]
    # only target
    y = df.iloc[:, 0]
    # apply balancing
    if balancing == True:
        # balance
        smt = SMOTE()
        X_balanced, y_balanced = smt.fit_resample(X, y)
        print(Counter(y_balanced))
        # save
        X = X_balanced
        y = y_balanced
    # return
    X, y
