import os
import sys
import numpy as np

# Calculate aggregate final results for an experiment
def aggregate_results(experiment_name, mode="Train"):
	path = "../../results/{}".format(experiment_name)
	number_of_runs = sum(1 for f in os.listdir("{}/tables".format(path)) if f.endswith('metrics.txt'))
	metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Loss']

	if mode == "Train":
		number_of_epochs = int(sum(1 for line in open("{}/tables/{}-1-metrics.txt".format(path, experiment_name), "r")) / 6)

		train_accuracies, train_precisions, train_recalls = np.zeros((number_of_runs)), np.zeros((number_of_runs)), np.zeros((number_of_runs))
		train_f1_scores, train_losses = np.zeros((number_of_runs)), np.zeros((number_of_runs))
		valid_accuracies, valid_precisions, valid_recalls = np.zeros((number_of_runs)), np.zeros((number_of_runs)), np.zeros((number_of_runs))
		valid_f1_scores, valid_losses = np.zeros((number_of_runs)), np.zeros((number_of_runs))

		train_metrics = np.zeros((number_of_runs, len(metrics)))
		valid_metrics = np.zeros((number_of_runs, len(metrics)))
		valid_aucs = np.zeros((number_of_runs, 3))

		# Fill up metrics arrays
		for i in range(number_of_runs):
			file = open("{}/tables/{}-{}-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()[-6:]

			for j in range(len(lines)):
				line = lines[j].split()
				if len(line) == 0:
					continue
				elif line[0] == "Train":
					train_metrics[i] = line[1:]
				elif line[0] == "Valid":
					valid_metrics[i] = line[1:]
				elif line[0] == "Class":
					valid_aucs[i] = line[-3:]

		# Calculate means
		train_accuracy_mean = round(np.mean(train_metrics[:,0]),2)
		train_precision_mean = round(np.mean(train_metrics[:,1]),2)
		train_recall_mean = round(np.mean(train_metrics[:,2]),2)
		train_f1_mean = round(np.mean(train_metrics[:,3]),2)
		train_loss_mean = round(np.mean(train_metrics[:,4]),4)

		valid_accuracy_mean = round(np.mean(valid_metrics[:,0]),2)
		valid_precision_mean = round(np.mean(valid_metrics[:,1]),2)
		valid_recall_mean = round(np.mean(valid_metrics[:,2]),2)
		valid_f1_mean = round(np.mean(valid_metrics[:,3]),2)
		valid_loss_mean = round(np.mean(valid_metrics[:,4]),4)

		germline_auc_mean = round(np.mean(valid_aucs[:,0]),4)
		somatic_auc_mean = round(np.mean(valid_aucs[:,1]),4)
		normal_auc_mean = round(np.mean(valid_aucs[:,2]),4)

		# Calculate standard errors
		train_accuracy_error = round(np.std(train_metrics[:,0]) / np.sqrt(number_of_runs),2)
		train_precision_error = round(np.std(train_metrics[:,1]) / np.sqrt(number_of_runs),2)
		train_recall_error = round(np.std(train_metrics[:,2]) / np.sqrt(number_of_runs),2)
		train_f1_error = round(np.std(train_metrics[:,3]) / np.sqrt(number_of_runs),2)
		train_loss_error = round(np.std(train_metrics[:,4]) / np.sqrt(number_of_runs),2)

		valid_accuracy_error = round(np.std(valid_metrics[:,0]) / np.sqrt(number_of_runs),2)
		valid_precision_error = round(np.std(valid_metrics[:,1]) / np.sqrt(number_of_runs),2)
		valid_recall_error = round(np.std(valid_metrics[:,2]) / np.sqrt(number_of_runs),2)
		valid_f1_error = round(np.std(valid_metrics[:,3]) / np.sqrt(number_of_runs),2)
		valid_loss_error = round(np.std(valid_metrics[:,4]) / np.sqrt(number_of_runs),2)

		germline_auc_error = round(np.std(valid_aucs[:,0]) / np.sqrt(number_of_runs),2)
		somatic_auc_error = round(np.std(valid_aucs[:,1]) / np.sqrt(number_of_runs),2)
		normal_auc_error = round(np.std(valid_aucs[:,2]) / np.sqrt(number_of_runs),2)

		# Save results
		results_file = open("{}/tables/aggregate_results.txt".format(path), "w")

		results_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\tLoss\n")
		results_file.write("Train\t{}+-{}\t\t".format(train_accuracy_mean, train_accuracy_error))
		results_file.write("{}+-{}\t\t".format(train_precision_mean, train_precision_error))
		results_file.write("{}+-{}\t\t".format(train_recall_mean, train_recall_error))
		results_file.write("{}+-{}\t\t".format(train_f1_mean, train_f1_error))
		results_file.write("{}+-{}\n".format(train_loss_mean, train_loss_error))
		
		results_file.write("Valid\t{}+-{}\t\t".format(valid_accuracy_mean, valid_accuracy_error))
		results_file.write("{}+-{}\t\t".format(valid_precision_mean, valid_precision_error))
		results_file.write("{}+-{}\t\t".format(valid_recall_mean, valid_recall_error))
		results_file.write("{}+-{}\t\t".format(valid_f1_mean, valid_f1_error))
		results_file.write("{}+-{}\n".format(valid_loss_mean, valid_loss_error))
		results_file.write("\n")

		results_file.write("Class Valid AUCs:\n")
		results_file.write("Germline Variant:\t{}+-{}\t\t".format(germline_auc_mean, germline_auc_error))
		results_file.write("Somatic Variant:\t{}+-{}\t\t".format(somatic_auc_mean, somatic_auc_error))
		results_file.write("Normal:\t{}+-{}".format(normal_auc_mean, normal_auc_error))

		results_file.close()

	elif mode == "Test":
		metrics = metrics[:-1]

		test_accuracies, test_precisions, test_recalls = np.zeros((number_of_runs)), np.zeros((number_of_runs)), np.zeros((number_of_runs))
		test_f1_scores, test_losses = np.zeros((number_of_runs)), np.zeros((number_of_runs))

		test_metrics = np.zeros((number_of_runs, len(metrics)))
		test_aucs = np.zeros((number_of_runs, 3))

		# Fill up metrics arrays
		for i in range(number_of_runs):
			file = open("{}/tables/test/{}-{}-test-metrics.txt".format(path, experiment_name, i+1), 'r')
			lines = file.readlines()

			for j in range(len(lines)):
				line = lines[j].split()
				if len(line) == 0:
					continue
				elif line[0] == "Test":
					test_metrics[i] = line[1:]
				elif line[0] == "Class":
					test_aucs[i] = line[-3:]

		# Calculate means
		test_accuracy_mean = round(np.mean(test_metrics[:,0]),2)
		test_precision_mean = round(np.mean(test_metrics[:,1]),2)
		test_recall_mean = round(np.mean(test_metrics[:,2]),2)
		test_f1_mean = round(np.mean(test_metrics[:,3]),2)

		germline_auc_mean = round(np.mean(test_aucs[:,0]),4)
		somatic_auc_mean = round(np.mean(test_aucs[:,1]),4)
		normal_auc_mean = round(np.mean(test_aucs[:,2]),4)

		# Calculate standard errors
		test_accuracy_error = round(np.std(test_metrics[:,0]) / np.sqrt(number_of_runs),2)
		test_precision_error = round(np.std(test_metrics[:,1]) / np.sqrt(number_of_runs),2)
		test_recall_error = round(np.std(test_metrics[:,2]) / np.sqrt(number_of_runs),2)
		test_f1_error = round(np.std(test_metrics[:,3]) / np.sqrt(number_of_runs),2)

		germline_auc_error = round(np.std(test_aucs[:,0]) / np.sqrt(number_of_runs),2)
		somatic_auc_error = round(np.std(test_aucs[:,1]) / np.sqrt(number_of_runs),2)
		normal_auc_error = round(np.std(test_aucs[:,2]) / np.sqrt(number_of_runs),2)

		# Save results
		results_file = open("{}/tables/test/aggregate_test_results.txt".format(path), "w")

		results_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\n")
		results_file.write("Test\t{}+-{}\t\t".format(test_accuracy_mean, test_accuracy_error))
		results_file.write("{}+-{}\t\t".format(test_precision_mean, test_precision_error))
		results_file.write("{}+-{}\t\t".format(test_recall_mean, test_recall_error))
		results_file.write("{}+-{}\t\t".format(test_f1_mean, test_f1_error))
		
		results_file.write("Class Valid AUCs:\n")
		results_file.write("Germline Variant:\t{}+-{}\t\t".format(germline_auc_mean, germline_auc_error))
		results_file.write("Somatic Variant:\t{}+-{}\t\t".format(somatic_auc_mean, somatic_auc_error))
		results_file.write("Normal:\t{}+-{}".format(normal_auc_mean, normal_auc_error))

		results_file.close()

# If run separately, the first command line argument is the experiment name
if __name__== "__main__":
	aggregate_results(sys.argv[1])