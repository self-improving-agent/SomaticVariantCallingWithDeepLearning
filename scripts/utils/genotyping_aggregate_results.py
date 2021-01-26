import os
import sys
import numpy as np

# Calculate aggregate final results for an experiment
def genotyping_aggregate_results(experiment_name, mode="Train"):
	path = "../../results/{}".format(experiment_name)
	number_of_runs = sum(1 for f in os.listdir("{}/tables".format(path)) if f.endswith('metrics.txt'))
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC', 'Loss']

	if mode == "Train":
		number_of_epochs = int(sum(1 for line in open("{}/tables/{}-1-metrics.txt".format(path, experiment_name), "r")) / 13)

		train_metrics = np.zeros((number_of_runs, len(metrics)))
		valid_metrics = np.zeros((number_of_runs, len(metrics)))

		# Fill up metrics arrays
		for i in range(number_of_runs):
			file = open("{}/tables/{}-{}-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()[-13:]

			for j in range(len(lines)):
				line = lines[j].split()
				if len(line) == 0:
					continue
				elif line[0] == "Train":
					if line[1] == "Accuracy":
						train_metrics[i, 0] = line[2]
					elif line[1] == "Precision":
						train_metrics[i, 1] = line[2]
					elif line[1] == "Recall":
						train_metrics[i, 2] = line[2]
					elif line[1] == "F1":
						train_metrics[i, 3] = line[3]
					elif line[1] == "Loss":
						train_metrics[i, 5] = line[2]
				elif line[0] == "Valid":
					if line[1] == "Accuracy":
						valid_metrics[i, 0] = line[2]
					elif line[1] == "Precision":
						valid_metrics[i, 1] = line[2]
					elif line[1] == "Recall":
						valid_metrics[i, 2] = line[2]
					elif line[1] == "F1":
						valid_metrics[i, 3] = line[3]
					elif line[1] == "AUC":
						valid_metrics[i, 4] = line[2]
					elif line[1] == "Loss":
						valid_metrics[i, 5] = line[2]
		# Calculate means
		train_accuracy_mean = np.round(np.mean(train_metrics[:,0]),2)
		train_precision_mean = np.round(np.mean(train_metrics[:,1]),2)
		train_recall_mean = np.round(np.mean(train_metrics[:,2]),2)
		train_f1_mean = np.round(np.mean(train_metrics[:,3]),2)
		train_loss_mean = np.round(np.mean(train_metrics[:,5]),4)

		valid_accuracy_mean = np.round(np.mean(valid_metrics[:,0], axis=0),2)
		valid_precision_mean = np.round(np.mean(valid_metrics[:,1], axis=0),2)
		valid_recall_mean = np.round(np.mean(valid_metrics[:,2], axis=0),2)
		valid_f1_mean = np.round(np.mean(valid_metrics[:,3], axis=0),2)
		valid_aucs_mean = np.round(np.mean(valid_metrics[:,4], axis=0),4)
		valid_loss_mean = np.round(np.mean(valid_metrics[:,5]),4)

		# Calculate standard errors
		train_accuracy_error = np.round(np.std(train_metrics[:,0]) / np.sqrt(number_of_runs),2)
		train_precision_error = np.round(np.std(train_metrics[:,1]) / np.sqrt(number_of_runs),2)
		train_recall_error = np.round(np.std(train_metrics[:,2]) / np.sqrt(number_of_runs),2)
		train_f1_error = np.round(np.std(train_metrics[:,3]) / np.sqrt(number_of_runs),2)
		train_loss_error = np.round(np.std(train_metrics[:,5]) / np.sqrt(number_of_runs),4)

		valid_accuracy_error = np.round(np.std(valid_metrics[:,0]) / np.sqrt(number_of_runs),2)
		valid_precision_error = np.round(np.std(valid_metrics[:,1]) / np.sqrt(number_of_runs),2)
		valid_recall_error = np.round(np.std(valid_metrics[:,2]) / np.sqrt(number_of_runs),2)
		valid_f1_error = np.round(np.std(valid_metrics[:,3]) / np.sqrt(number_of_runs),2)
		valid_aucs_error = np.round(np.std(valid_metrics[:,4]) / np.sqrt(number_of_runs),4)
		valid_loss_error = np.round(np.std(valid_metrics[:,5]) / np.sqrt(number_of_runs),4)

		# Save results
		results_file = open("{}/tables/aggregate_results.txt".format(path), "w")

		results_file.write("Train Loss\t\t{}+-{}\n".format(train_loss_mean, train_loss_error))
		results_file.write("Valid Loss\t\t{}+-{}\n".format(valid_loss_mean, valid_loss_error))		
		results_file.write("Train Accuracy\t\t{}+-{}\n".format(train_accuracy_mean, train_accuracy_error))
		results_file.write("Train Precision\t{}+-{}\n".format(train_precision_mean, train_precision_error))
		results_file.write("Train Recall\t\t{}+-{}\n".format(train_recall_mean, train_recall_error))
		results_file.write("Train F1 Score\t\t{}+-{}\n".format(train_f1_mean, train_f1_error))
		results_file.write("Valid Accuracy\t\t{}+-{}\n".format(valid_accuracy_mean, valid_accuracy_error))
		results_file.write("Valid Precision\t{}+-{}\n".format(valid_precision_mean, valid_precision_error))
		results_file.write("Valid Recall\t\t{}+-{}\n".format(valid_recall_mean, valid_recall_error))
		results_file.write("Valid F1 Score\t\t{}+-{}\n".format(valid_f1_mean, valid_f1_error))
		results_file.write("Valid AUC\t\t{}+-{}\n".format(valid_aucs_mean, valid_aucs_error))
		results_file.close()

		results_file.close()

	elif mode == "Test":
		metrics = metrics[:-1]

		test_metrics = np.zeros((number_of_runs, len(metrics)))

		# Fill up metrics arrays
		for i in range(number_of_runs):
			file = open("{}/tables/test/{}-{}-test-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()
				if len(line) == 0:
					continue
				elif line[1] == "Accuracy":
						test_metrics[i, 0] = line[2]
				elif line[1] == "Precision":
					test_metrics[i, 1] = line[2]
				elif line[1] == "Recall":
					test_metrics[i, 2] = line[2]
				elif line[1] == "F1":
					test_metrics[i, 3] = line[3]
				elif line[1] == "AUC":
					test_metrics[i, 4] = line[2]

		# Calculate means
		test_accuracy_mean = np.round(np.mean(test_metrics[:,0]),2)
		test_precision_mean = np.round(np.mean(test_metrics[:,1]),2)
		test_recall_mean = np.round(np.mean(test_metrics[:,2]),2)
		test_f1_mean = np.round(np.mean(test_metrics[:,3]),2)
		test_aucs_mean = np.round(np.mean(test_metrics[:,4]),4)

		# Calculate standard errors
		test_accuracy_error = np.round(np.std(test_metrics[:,0]) / np.sqrt(number_of_runs),2)
		test_precision_error = np.round(np.std(test_metrics[:,1]) / np.sqrt(number_of_runs),2)
		test_recall_error = np.round(np.std(test_metrics[:,2]) / np.sqrt(number_of_runs),2)
		test_f1_error = np.round(np.std(test_metrics[:,3]) / np.sqrt(number_of_runs),2)
		test_aucs_error = np.round(np.std(test_metrics[:,4]) / np.sqrt(number_of_runs),4)

		# Save results
		results_file = open("{}/tables/test/aggregate_test_results.txt".format(path), "w")

		results_file.write("Test Accuracy\t\t{}+-{}\n".format(test_accuracy_mean, test_accuracy_error))
		results_file.write("Test Precision\t\t{}+-{}\n".format(test_precision_mean, test_precision_error))
		results_file.write("Test Recall\t\t{}+-{}\n".format(test_recall_mean, test_recall_error))
		results_file.write("Test F1 Score\t\t{}+-{}\n".format(test_f1_mean, test_f1_error))
		results_file.write("Test AUC\t\t{}+-{}\n".format(test_aucs_mean, test_aucs_error))
		results_file.close()

# If run separately, the first command line argument is the experiment name
if __name__== "__main__":
	aggregate_results(sys.argv[1])