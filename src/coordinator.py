""" Manage All the Program """

# imports
import timeit
from classification import classify, definePaths
from os import listdir, path
from preprocessing import readDataset
from classification import runNN

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
        X, y = readDataset(info[0],True)
        # run neural networks for multiple configurations
        # runNN(X, y, config, k_fold, random_state)



    # # provide paths
    # definePaths(mainInputPath, mainOutputPath)
    # # CART
    # classify('CART')
    # # ID3
    # classify('ID3')
    # # C45
    # classify('C4.5')
    # # time end
    # elapsed = timeit.default_timer()-tic
    # print(f'Time elapsed is aproximately {elapsed} seconds o {elapsed/60} minutes')  # seconds
