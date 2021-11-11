""" Manage All the Program """

# imports
import timeit
from classification import classify, definePaths
from os import listdir, path
from preprocessing import getTopInfo, readDataset, formatTable
from classification import runNN
from graph import drawTable, graphCurve

# input paths
baseInPath = r'data\input'

# output paths
baseOutPath = r'data\output'


# Coordinator
def coordinate(config, k_fold, random_state):
    # time init
    tic = timeit.default_timer()
    # paths and names for hypothesis
    fileNames = [f.replace('.csv', '') for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    filePaths = [path.join(baseInPath, f) for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    fileInformation = zip(fileNames, filePaths)
    # loop hypothesis
    for info in fileInformation:
        # read dataset and apply balancing
        X, y = readDataset(info[1],True)
        # run neural networks for multiple configurations
        all_results = runNN(X, y, config, k_fold, random_state)
        # ordering
        table_results_ordered = formatTable(all_results, True)
        # table_results_not_ordered = formatTable(results, False)
        # table to compare models
        drawTable(table_results_ordered, info[0], baseOutPath)
        # get top result classification vectors
        topRow = table_results_ordered[0]
        y_true, y_pred_proba, y_pred_categ, loss_epoch = getTopInfo(topRow, all_results)
        print(y_true)
        print(y_pred_proba)
        print(y_pred_categ)
        print(loss_epoch)
        # graphs
        graphs = ['ROC', 'PrecRecall', 'confMatrix', 'lossEpoch']
        # apply graphs
        for graphName in graphs:
            graphCurve(y_true, y_pred_proba, graphName, info[0], baseOutPath, y_pred_categ, loss_epoch)
