import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# Produce plots for a given experiment
def produce_plots(experiment_name, mode="Train"):
	path = "../../results/{}".format(experiment_name)
	number_of_runs = sum(1 for f in os.listdir("{}/tables".format(path)) if f.endswith('metrics.txt'))
	number_of_epochs = int(sum(1 for line in open("{}/tables/{}-1-metrics.txt".format(path, experiment_name), "r")) / 14)
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', "Loss"]

	if mode == "Train":
		train_data = np.zeros((number_of_runs, number_of_epochs, len(metrics), 3))
		valid_data = np.zeros((number_of_runs, number_of_epochs, len(metrics), 3))

		# first_words_0 = []
		# first_words_10 = []

		# Fill up data arrays
		for i in range(number_of_runs):
			file = open("{}/tables/{}-{}-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()	
				epoch = int(j / 14)

				if len(line) == 0:
					continue
				elif line[0] == "Train":
					if line[1] == "Accuracy":
						train_data[i, epoch, 0] = line[2:]
					elif line[1] == "Precision":
						train_data[i, epoch, 1] = line[2:]
					elif line[1] == "Recall":
						train_data[i, epoch, 2] = line[2:]
					elif line[1] == "F1":
						train_data[i, epoch, 3] = line[3:]
					elif line[1] == "Loss":
						train_data[i, epoch, 5, 0] = line[2]
				elif line[0] == "Valid":
					if line[1] == "Accuracy":
						valid_data[i, epoch, 0] = line[2:]
					elif line[1] == "Precision":
						valid_data[i, epoch, 1] = line[2:]
					elif line[1] == "Recall":
						valid_data[i, epoch, 2] = line[2:]
					elif line[1] == "F1":
						valid_data[i, epoch, 3] = line[3:]
					elif line[1] == "AUCs":
						valid_data[i, epoch, 4] = line[2:]
					elif line[1] == "Loss":
						valid_data[i, epoch, 5, 0] = line[2]

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
			if metrics[i] == "Loss":
				df1 = pd.DataFrame(train_data[:,:,i,0]).melt()
				df2 = pd.DataFrame(valid_data[:,:,i,0]).melt()

				df1['Dataset'] = 'Train'
				df2['Dataset'] = 'Valid'

				df = pd.concat([df1,df2]).rename(columns={'variable': 'Epoch', 'value': metrics[i]})

				sns.lineplot(x='Epoch', y=metrics[i], hue='Dataset', data=df)
				plt.legend(loc="best")
				plt.grid("on")
				plt.savefig("{}/figures/{}.pdf".format(path, metrics[i]))
				plt.clf()

			else:
				# Train set plots
				df1 = pd.DataFrame(train_data[:,:,i,0]).melt()
				df2 = pd.DataFrame(train_data[:,:,i,1]).melt()
				df3 = pd.DataFrame(train_data[:,:,i,2]).melt()

				df1['Class'] = 'Germline Variant'
				df2['Class'] = 'Somatic Variant'
				df3['Class'] = 'Normal' 
				
				df = pd.concat([df1,df2,df3]).rename(columns={'variable': 'Epoch', 'value': metrics[i]})

				sns.lineplot(x='Epoch', y=metrics[i], hue='Class', data=df)
				plt.legend(loc="best")
				plt.grid("on")
				plt.savefig("{}/figures/train/{}.pdf".format(path, metrics[i]))
				plt.clf()

				# Validation set plots 
				df1 = pd.DataFrame(valid_data[:,:,i,0]).melt()
				df2 = pd.DataFrame(valid_data[:,:,i,1]).melt()
				df3 = pd.DataFrame(valid_data[:,:,i,2]).melt()

				df1['Class'] = 'Germline Variant'
				df2['Class'] = 'Somatic Variant'
				df3['Class'] = 'Normal' 

				df = pd.concat([df1,df2,df3]).rename(columns={'variable': 'Epoch', 'value': metrics[i]})

				sns.lineplot(x='Epoch', y=metrics[i], hue='Class', data=df)
				plt.legend(loc="best")
				plt.grid("on")
				plt.savefig("{}/figures/valid/{}.pdf".format(path, metrics[i]))
				plt.clf()

		os.remove("{}/figures/train/AUC.pdf".format(path, metrics[i]))

	elif mode == "Test":
		number_of_saves = int(sum(1 for line in open("{}/tables/test/{}-1-test-metrics.txt".format(path, experiment_name), "r")) / 8)
		metrics = metrics[:-1]

		test_data = np.zeros((number_of_runs, number_of_saves, len(metrics), 3))

		# Fill up data arrays
		for i in range(number_of_runs):
			file = open("{}/tables/test/{}-{}-test-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()	
				save = int(j / 8)

				if len(line) == 0:
					continue
				elif line[0] == "Test":
					if line[1] == "Accuracy":
						test_data[i, save, 0] = line[2:]
					elif line[1] == "Precision":
						test_data[i, save, 1] = line[2:]
					elif line[1] == "Recall":
						test_data[i, save, 2] = line[2:]
					elif line[1] == "F1":
						test_data[i, save, 3] = line[3:]
					elif line[1] == "AUCs":
						test_data[i, save, 4] = line[2:]

		# Create plots
		for i in range(len(metrics)):
			# Test set plots
			df1 = pd.DataFrame(test_data[:,:,i,0]).melt()
			df2 = pd.DataFrame(test_data[:,:,i,1]).melt()
			df3 = pd.DataFrame(test_data[:,:,i,2]).melt()

			df1['Class'] = 'Germline Variant'
			df2['Class'] = 'Somatic Variant'
			df3['Class'] = 'Normal' 
			
			df = pd.concat([df1,df2,df3]).rename(columns={'variable': 'Save', 'value': metrics[i]})

			sns.lineplot(x='Save', y=metrics[i], hue='Class', data=df)
			plt.legend(loc="best")
			plt.grid("on")
			plt.savefig("{}/figures/test/{}.pdf".format(path, metrics[i]))
			plt.clf()

# If run separately, the first command line argument is the experiment name
if __name__== "__main__":
	produce_plots(sys.argv[1])