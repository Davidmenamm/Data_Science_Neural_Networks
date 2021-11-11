""" For all graphing functions """

# imports
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import metrics
from matplotlib.widgets import Slider


# graph auc
def graphCurve(y_true, y_pred_proba, functionName, fileName, basePath, y_pred_categ, loss_epoch):
	# base folder
	baseFolder = basePath+f'\_{fileName}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_TopResult_{functionName}.png'
	# give values
	fig = plt.figure(f'{fileName}-TopResult-{functionName}')
	plt.title(f'{fileName} - {functionName}')
	# manage cases
	graph = None
	if functionName == 'ROC':
		graph = metrics.RocCurveDisplay.from_predictions(y_true, y_pred_proba, name='TopResult', ax=plt.gca())
	elif functionName == 'PrecRecall':
		graph = metrics.PrecisionRecallDisplay.from_predictions(y_true, y_pred_proba, name='TopResult', ax=plt.gca())
	elif functionName == 'confMatrix':
		graph = metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred_categ, ax=plt.gca())
	elif functionName == 'lossEpoch':
		# epoch
		x_axis = range(1, len(loss_epoch)+1)
		# loss
		y_axis = loss_epoch
		# figure, axes
		plt.plot(x_axis, y_axis, 'bo-', label = 'loss vs epochs')
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		# plt.legend(loc='best')
	# plot
	fig.savefig(filePath, dpi=fig.dpi)
	plt.clf()






# draw table
def drawTable(data, fileName, basePath):
	print(data)
	# base folder
	baseFolder = basePath+f'\_{fileName}'
	# check if already exists
	if not os.path.exists(baseFolder):
		os.makedirs(baseFolder)
	# algorithm graphs
	filePath = f'{baseFolder}\\_table.png'
	# get column names
	columns = ['Layers', 'Active', 'lr', 'epoch',
				'acc_avg', 'acc_std', 'prec_avg',
				'prec_std','rec_avg', 'rec_std',
				'auc_avg', 'auc_std','loss_avg',
				'loss_std']
	# counter	
	count = 0
	# get values
	cell_text = []
	for row in data:
		rowStr = []
		for value in row:
			# append
			rowStr.append(value)
		# append
		cell_text.append(rowStr)
		# increment
		count += 1
	# figure
	fig, ax = plt.subplots()
	fig.set_size_inches(20, 61)
	# hide axes
	fig.patch.set_visible(False)
	ax.axis('off')
	ax.axis('tight')
	# axis content
	print(count)
	labels_vertical = [str(lbl) for lbl in range(1, count+1)]
	ytable = ax.table(cellText = cell_text, rowLabels = labels_vertical, colLabels=columns, loc='center')
	ytable.set_fontsize(24)
	ytable.scale(1, 4)
	# fig.tight_layout()
	# plot
	fig.savefig(filePath, dpi=fig.dpi)
	fig.clf()