import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# Produce plots for a given experiment
def genotyping_produce_plots(experiment_name, mode="Train"):
	path = "../../results/{}".format(experiment_name)
	number_of_runs = sum(1 for f in os.listdir("{}/tables".format(path)) if f.endswith('metrics.txt'))
	number_of_epochs = int(sum(1 for line in open("{}/tables/{}-1-metrics.txt".format(path, experiment_name), "r")) / 13)
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Loss']

	if mode == "Train":
		train_data = np.zeros((number_of_runs, number_of_epochs, len(metrics)))
		valid_data = np.zeros((number_of_runs, number_of_epochs, len(metrics)))

		# first_words_0 = []
		# first_words_5 = []

		# Fill up data arrays
		for i in range(number_of_runs):
			file = open("{}/tables/{}-{}-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()	
				epoch = int(j / 13)

				if len(line) == 0:
					continue
				elif line[0] == "Train":
					if line[1] == "Accuracy":
						train_data[i, epoch, 0] = line[2]
					elif line[1] == "Precision":
						train_data[i, epoch, 1] = line[2]
					elif line[1] == "Recall":
						train_data[i, epoch, 2] = line[2]
					elif line[1] == "F1":
						train_data[i, epoch, 3] = line[3]
					elif line[1] == "Loss":
						train_data[i, epoch, 5] = line[2]
				elif line[0] == "Valid":
					if line[1] == "Accuracy":
						valid_data[i, epoch, 0] = line[2]
					elif line[1] == "Precision":
						valid_data[i, epoch, 1] = line[2]
					elif line[1] == "Recall":
						valid_data[i, epoch, 2] = line[2]
					elif line[1] == "F1":
						valid_data[i, epoch, 3] = line[3]
					elif line[1] == "AUC":
						valid_data[i, epoch, 4] = line[2]
					elif line[1] == "Loss":
						valid_data[i, epoch, 5] = line[2]

		# # Uncomment these when need to identify corrupted metrics file
		#
		# 		if i == 0:
		# 			first_words_0.append(line[0])
		# 		elif i == 4:
		# 			first_words_5.append(line[0])

		# print(len(first_words_0))
		# print(len(first_words_5))

		# for i in range(len(first_words_5)):
		# 	if first_words_0[i] != first_words_5[i]:
		# 		print(first_words_5[i], first_words_0[i])
		# 		print(i)
		# 		break

		# print(error)


		# Create plots
		for i in range(len(metrics)):
			if metrics[i] == "AUC":
				df = pd.DataFrame(valid_data[:,:,i]).melt().rename(columns={'variable': 'Epoch', 'value': metrics[i]})
				df['Dataset'] = 'Valid'
				sns.lineplot(x='Epoch', y=metrics[i], hue='Dataset', data=df)
				plt.legend(loc="best")
				plt.grid("on")
				plt.savefig("{}/figures/{}.pdf".format(path, metrics[i]))
				plt.clf()

			else:
				df1 = pd.DataFrame(train_data[:,:,i]).melt()
				df2 = pd.DataFrame(valid_data[:,:,i]).melt()

				df1['Dataset'] = 'Train'
				df2['Dataset'] = 'Valid'

				df = pd.concat([df1,df2]).rename(columns={'variable': 'Epoch', 'value': metrics[i]})

				sns.lineplot(x='Epoch', y=metrics[i], hue='Dataset', data=df)
				plt.legend(loc="best")
				plt.grid("on")
				plt.savefig("{}/figures/{}.pdf".format(path, metrics[i]))
				plt.clf()

	elif mode == "Test":
		number_of_saves = int(sum(1 for line in open("{}/tables/test/{}-1-test-metrics.txt".format(path, experiment_name), "r")) / 7)
		metrics = metrics[:-1]

		test_data = np.zeros((number_of_runs, number_of_saves, len(metrics)))

		# Fill up data arrays
		for i in range(number_of_runs):
			file = open("{}/tables/test/{}-{}-test-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()	
				save = int(j / 7)

				if len(line) == 0:
					continue
				elif line[0] == "Test":
					if line[1] == "Accuracy":
						test_data[i, save, 0] = line[2]
					elif line[1] == "Precision":
						test_data[i, save, 1] = line[2]
					elif line[1] == "Recall":
						test_data[i, save, 2] = line[2]
					elif line[1] == "F1":
						test_data[i, save, 3] = line[3]
					elif line[1] == "AUC":
						test_data[i, save, 4] = line[2]

		# Create plots
		for i in range(len(metrics)):
			# Test set plots
			df = pd.DataFrame(test_data[:,:,i]).melt().rename(columns={'variable': 'Save', 'value': metrics[i]})
			df['Dataset'] = 'Test'
			sns.lineplot(x='Save', y=metrics[i], hue='Dataset', data=df)
			plt.legend(loc="best")
			plt.grid("on")
			plt.savefig("{}/figures/Test_{}.pdf".format(path, metrics[i]))
			plt.clf()

# If run separately, the first command line argument is the experiment name
if __name__== "__main__":
	produce_plots(sys.argv[1])