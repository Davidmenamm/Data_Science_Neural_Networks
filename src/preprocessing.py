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
    print(inPath)
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
    return X, y



# ordering based on the best auc
def formatTable(data, ordered):
    # get values
    table = []
    idx_auc = 0
    for topology in data:
        for config in topology:
            row = []
            # add model info
            row.extend( [str(info) for info in config['model']] )
            # add results
            row.append(str(config['acc_avg']))
            row.append(str(config['acc_std']))
            row.append(str(config['prec_avg']))
            row.append(str(config['prec_std']))
            row.append(str(config['rec_avg']))
            row.append(str(config['rec_std']))
            row.append(str(config['auc_avg']))
            row.append(str(config['auc_std']))
            row.append(str(config['loss_avg']))
            row.append(str(config['loss_std']))
            # index where auc_avg is located among all lists
            if idx_auc == 0: row.index(str(config['auc_avg']))
            # append to table data
            table.append(row)
    # order table
    if sorted:
        table = sorted(table, key= lambda item : item[idx_auc], reverse=True)
    # return
    return table



# get top info
def getTopInfo(topConfig, data):
    # get classification vectors
    y_pred_true = []
    y_pred_proba = []
    y_pred_categ = []
    loss_epoch = []
    # iterate topologies
    for topology in data:
        # iterate hyperparams
        for config in topology:
            # top config identifier
            topConfigId = []
            # first for elements identify each topology
            topConfigId.append(topConfig[0])
            topConfigId.append(topConfig[1])
            topConfigId.append(topConfig[2])
            topConfigId.append(topConfig[3])
            print(topConfigId)
            print(config['model'])
            # find top model
            validate = True
            for identifier in config['model']:
                if str(identifier) not in topConfigId:
                    validate = False
            if validate:
                print('validate')
                y_pred_true = config['y_true_sum']
                y_pred_proba = config['y_pred_proba_sum']
                y_pred_categ = config['y_pred_categ_sum']
                loss_epoch = config['loss_epoch_avg']
    # return
    return y_pred_true, y_pred_proba, y_pred_categ, loss_epoch