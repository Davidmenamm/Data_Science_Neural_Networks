""" Apply binary classification with ANN """

# imports
import pandas as pd
import itertools as itl
import numpy as np
from os import listdir, path
from sklearn import model_selection as ms, neural_network as nn, metrics as mtc
from graph import graphCurve
from imblearn.over_sampling import SMOTE
from collections import Counter


# run one or multiple NN configuration(s)
def runNN(X, y, config, k_fold, random_state):
    # numerosity or tuples
    numerosity = len(X)
    # run for all topologies
    topology_results = []
    for topology in config['topologies']:
        # run all hyper parameters
        hyperparam_results = []
        # hyper parameters
        cartesian = itl.product(config['hyperparameter']['lr'], config['hyperparameter']['epoch'])
        for lr, epoch in cartesian:
            # stratified cross validation
            skf = ms.StratifiedKFold(n_splits=k_fold, random_state=random_state, shuffle=True)
            stratified_indices = skf.split(X, y)
            # config results
            configuration_results = []
            # run all k fold
            for trainIndices, testIndices in stratified_indices:
                # train and test
                X_train, X_test = X.iloc[trainIndices, :], X.iloc[testIndices, :]
                y_train, y_test = y.iloc[trainIndices], y.iloc[testIndices]
                # neural network definition
                clf = nn.MLPClassifier(
                    hidden_layer_sizes = topology['layers'],
                    activation = topology['activation'],
                    solver = 'sgd',
                    batch_size = numerosity,
                    learning_rate = 'constant',
                    learning_rate_init = lr,
                    max_iter = epoch,
                    shuffle = False,
                    random_state = random_state,
                    momentum = 0,
                    n_iter_no_change = epoch, 
                    # verbose=True
                )
                # classify
                clf.fit(X_train,y_train)
                y_pred_proba = [ prob[1] for prob in clf.predict_proba(X_test) ]
                y_pred_categ = list( map(lambda a : 1 if a > 0.5 else 0, y_pred_proba) )
                # metrics
                acc = mtc.accuracy_score(y_test.tolist(), y_pred_categ)
                prec = mtc.precision_score(y_test.tolist(), y_pred_categ)
                rec = mtc.recall_score(y_test.tolist(), y_pred_categ)
                auc = mtc.roc_auc_score(y_test.tolist(), y_pred_categ)
                tn, fp, fn, tp = mtc.confusion_matrix(y_test.tolist(), y_pred_categ).ravel()
                loss = clf.loss_
                loss_epoch = clf.loss_curve_
                # save specific configuration metrics
                metrics= {}
                metrics['acc'] = acc
                metrics['prec'] = prec
                metrics['rec'] = rec
                metrics['auc'] = auc
                metrics['tn'] = tn
                metrics['fp'] = fp
                metrics['fn'] = fn
                metrics['tp'] = tp
                metrics['loss'] = loss
                metrics['loss_epoch'] = loss_epoch
                metrics['y_pred_categ'] = y_pred_categ
                metrics['y_pred_proba'] = y_pred_proba
                metrics['y_true'] = y_test.tolist()
                # append
                configuration_results.append(metrics)
                print(clf.out_activation_)
            # hyperparam metrics
            accs = [ conf['acc'] for conf in configuration_results ]
            precs = [ conf['prec'] for conf in configuration_results ]
            recs = [ conf['rec'] for conf in configuration_results ]
            aucs = [ conf['acc'] for conf in configuration_results ]
            losses = [ conf['loss'] for conf in configuration_results ]
            # conf matrix
            tns = [ conf['tn'] for conf in configuration_results ]
            fps = [ conf['fp'] for conf in configuration_results ]
            fns = [ conf['fn'] for conf in configuration_results ]
            tps = [ conf['tp'] for conf in configuration_results ]
            # loss
            loss_epochs = [ conf['loss_epoch'] for conf in configuration_results ]
            y_pred_categs = [ conf['y_pred_categ'] for conf in configuration_results ]
            y_pred_probas = [ conf['y_pred_proba'] for conf in configuration_results ]
            y_trues = [ conf['y_true'] for conf in configuration_results ]
            # sum
            tn_sum = sum(tns)
            fp_sum = sum(fps)
            fn_sum = sum(fns)
            tp_sum = sum(tps)
            # avg
            acc_avg = np.mean(accs)
            prec_avg = np.mean(precs)
            rec_avg = np.mean(recs)
            auc_avg = np.mean(aucs)
            loss_avg = np.mean(losses)
            # std
            acc_std = np.std(accs)
            prec_std = np.std(precs)
            rec_std = np.std(recs)
            auc_std = np.std(aucs)
            loss_std = np.std(losses)
            # sum of lists
            y_pred_categ_sum = []
            y_pred_proba_sum = []
            y_true_sum = [] 
            for item in y_pred_categs: y_pred_categ_sum.extend(item)
            for item in y_pred_probas: y_pred_proba_sum.extend(item)
            for item in y_trues: y_true_sum.extend(item)
            # avg of lists
            loss_epoch_sum = []
            loss_epoch_len = len(loss_epochs)                   
            for idx, item in enumerate(loss_epochs):
                if idx == 0:
                    loss_epoch_sum.extend(item)
                else:
                    loss_epoch_sum = [ x+y for x,y in zip(loss_epoch_sum, item)]
            loss_epoch_avg = [value/loss_epoch_len for value in loss_epoch_sum ]
                    
            # save results
            metrics = {}
            metrics['model'] = [topology['layers'], topology['activation'], f'{lr}', f'{epoch}']
            metrics['tn_sum'] = tn_sum
            metrics['fp_sum'] = fp_sum
            metrics['fn_sum'] = fn_sum
            metrics['tp_sum'] = tp_sum
            metrics['acc_avg'] = round(acc_avg, 2)
            metrics['prec_avg'] = round(prec_avg, 2)
            metrics['rec_avg'] = round(rec_avg, 2)
            metrics['auc_avg'] = round(auc_avg, 2)
            metrics['loss_avg'] = round(loss_avg, 2)
            metrics['acc_std'] = round(acc_std, 2)
            metrics['prec_std'] = round(prec_std, 2)
            metrics['rec_std'] = round(rec_std, 2)
            metrics['auc_std'] = round(auc_std, 2)
            metrics['loss_std'] = round(loss_std, 2)
            metrics['loss_epoch_avg'] = loss_epoch_avg
            metrics['y_pred_categ_sum'] = y_pred_categ_sum
            metrics['y_pred_proba_sum'] = y_pred_proba_sum
            metrics['y_true_sum'] = y_true_sum
            # append
            hyperparam_results.append(metrics)
        # save topologies
        topology_results.append(hyperparam_results)
    # return
    return topology_results        







# define paths
def definePaths(inputPath, outputPath):
    definePaths.baseInPath = inputPath
    definePaths.baseOutPath = outputPath


# classify
def classify(algorithm, nFolds=10, randomSeed=100):
    # find input files
    baseInPath = definePaths.baseInPath
    fileNames = [f.replace('.csv', '') for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    filePaths = [path.join(baseInPath, f) for f in listdir(
        baseInPath) if path.isfile(path.join(baseInPath, f))]
    fileInformation = zip(fileNames, filePaths)
    # loop files
    for info in fileInformation:
        # read csv
        df = pd.read_csv(info[1], engine='c')
        # only features
        X_imbalance = df.iloc[:, 1: len(df.columns)]
        # only target
        y_imbalance = df.iloc[:, 0]
        # balance classes with sampling
        smt = SMOTE()
        X, y = smt.fit_resample(X_imbalance, y_imbalance)
        print(Counter(y))
        # stratified cross validation
        skf = ms.StratifiedKFold(
            n_splits=nFolds, random_state=randomSeed, shuffle=True)
        stratified_indices = skf.split(X, y)
        # results
        results = dict()
        # stratify k-fold cross
        count = 0
        for trainIndices, testIndices in stratified_indices:
            X_train, X_test = X.iloc[trainIndices, :], X.iloc[testIndices, :]
            y_train, y_test = y.iloc[trainIndices], y.iloc[testIndices]
            # apply selected classification
            infoTree = run_tree(X_train, X_test, y_train, y_test, algorithm)
            # join metrics for all cross validations, for each tree
            # initial values
            if(len(results)==0):
                # list extending
                results['y_true'] = []
                results['y_pred_categ_list'] = []
                results['y_pred_prob_list'] = []
                # scalar sum
                results['tn'] = 0
                results['fp'] = 0
                results['fn'] = 0
                results['tp'] = 0
                # scalar avg
                results['acc'] = []
                results['prec'] = []
                results['rec'] = []
                results['auc'] = []
            else:
                # target
                results['y_true'].extend(infoTree['y_true'])
                results['y_pred_categ_list'].extend(infoTree['y_pred_categ_list'])
                results['y_pred_prob_list'].extend(infoTree['y_pred_prob_list'])
                # confusion matrix
                results['tn'] += infoTree['tn']
                results['fp'] += infoTree['fp']
                results['fn'] += infoTree['fn']
                results['tp'] += infoTree['tp']
                # metrics avg
                results['acc'].append(infoTree['acc'])
                results['prec'].append(infoTree['prec'])
                results['rec'].append(infoTree['rec'])
                results['auc'].append(infoTree['auc'])
            # add counter
            count += 1
        # avg metrics
        acc_np = np.array(results['acc'])
        prec_np = np.array(results['prec'])
        rec_np = np.array(results['rec'])
        auc_np = np.array(results['auc'])
        # mean
        results['acc_avg'] = np.mean(acc_np)
        results['prec_avg'] = np.mean(prec_np)
        results['rec_avg'] = np.mean(rec_np)
        results['auc_avg'] = np.mean(auc_np)
        # std
        results['acc_std'] = np.std(acc_np)
        results['prec_std'] = np.std(prec_np)
        results['rec_std'] = np.std(rec_np)
        results['auc_std'] = np.std(auc_np)
        # write results to file
        writeTextFile(info[0], definePaths.baseOutPath, results, algorithm)
        # Graphics
        plotName = f'{info[0]}'
        function1 = 'ROC'
        function2 = 'PrecRecall'
        function3= 'confMatrix'
        # auc
        graphCurve(results['y_true'], results['y_pred_prob_list'], algorithm, function1, info[0], definePaths.baseOutPath)
        # prec vs recall
        graphCurve(results['y_true'], results['y_pred_prob_list'], algorithm, function2, info[0], definePaths.baseOutPath)
        # conf matrix
        graphCurve(results['y_true'], results['y_pred_prob_list'], algorithm, function3, info[0], definePaths.baseOutPath, results['y_pred_categ_list'])