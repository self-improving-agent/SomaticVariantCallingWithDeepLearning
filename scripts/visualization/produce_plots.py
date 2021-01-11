import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# Produce plots for a given experiment
def produce_plots(experiment_name):
	path = "../../results/{}".format(experiment_name)
	number_of_runs = sum(1 for f in os.listdir("{}/tables".format(path)) if f.endswith('metrics.txt'))
	number_of_epochs = int(sum(1 for line in open("{}/tables/{}-1-metrics.txt".format(path, experiment_name), "r")) / 6)
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss']

	train_data = np.zeros((number_of_runs, number_of_epochs, len(metrics)))
	valid_data = np.zeros((number_of_runs, number_of_epochs, len(metrics)))
	valid_aucs_data = np.zeros((number_of_runs, number_of_epochs, 3))

	# first_words_0 = []
	# first_words_10 = []

	# Fill up data arrays
	for i in range(number_of_runs):
		file = open("{}/tables/{}-{}-metrics.txt".format(path, experiment_name, i+1), 'r')
		lines = file.readlines()

		for j in range(len(lines)):
			line = lines[j].split()	
			epoch = int(j / 6)

			if len(line) == 0:
				continue
			elif line[0] == "Train":
				train_data[i, epoch] = line[1:]
			elif line[0] == "Valid":
				valid_data[i, epoch] = line[1:]
			elif line[0] == "Class":
				valid_aucs_data[i, epoch] = line[-3:]

	# # Uncomment these when need to identify corrupted metrics file
	#
	# 		if i == 0:
	# 			first_words_0.append(line[0])
	# 		elif i == 9:
	# 			first_words_10.append(line[0])

	# print(len(first_words_0))
	# print(len(first_words_10))

	# for i in range(len(first_words_10)):
	# 	if first_words_0[i] != first_words_10[i]:
	# 		print(first_words_10[i])
	# 		print(i)
	# 		break


	# Create plots
	for i in range(len(metrics)):
		df1 = pd.DataFrame(train_data[:,:,i]).melt()
		df1['Dataset'] = 'Train'
		df2 = pd.DataFrame(valid_data[:,:,i]).melt()
		df2['Dataset'] = 'Valid'
		df = pd.concat([df1,df2]).rename(columns={'variable': 'Epoch', 'value': metrics[i]})

		sns.lineplot(x='Epoch', y=metrics[i], hue='Dataset', data=df)
		plt.legend(loc="best")
		plt.grid("on")
		plt.savefig("{}/figures/{}.pdf".format(path, metrics[i]))
		plt.clf()

	# Create AUC plot separately
	df1 = pd.DataFrame(valid_aucs_data[:,:,0]).melt()
	df1['Class'] = 'Germline Variant'
	df2 = pd.DataFrame(valid_aucs_data[:,:,1]).melt()
	df2['Class'] = 'Somatic Variant'
	df3 = pd.DataFrame(valid_aucs_data[:,:,2]).melt()
	df3['Class'] = 'Normal'
	df = pd.concat([df1,df2, df3]).rename(columns={'variable': 'Epoch', 'value': "AUC"})

	sns.lineplot(x='Epoch', y='AUC', hue='Class', data=df)
	plt.legend(loc="best")
	plt.grid("on")
	plt.savefig("{}/figures/AUC.pdf".format(path))
	plt.clf()

# If run separately, the first command line argument is the experiment name
if __name__== "__main__":
	produce_plots(sys.argv[1])