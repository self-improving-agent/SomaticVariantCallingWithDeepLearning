import numpy as np
import matplotlib.pyplot as plt
import os

from model_training import train_model

NUMBER_OF_RUNS = snakemake.config["number_of_runs"]
EXPERIMENT_NAME = snakemake.config["experiment_name"]
MODEL_TYPE = snakemake.config["model_type"]

# Hyperparameters
EPOCHS = snakemake.config["epochs"]
BATCH_SIZE = snakemake.config["batch_size"]
LEARNING_RATE = snakemake.config["learnin_rate"]
HIDDEN_UNITS = snakemake.config["hidden_units"]
LAYERS = snakemake.config["hidden_layers"]
DROPOUT = snakemake.config["dropout"] # Set to positive to include dropout layers
BIDIRECTIONAL = snakemake.config["bidirectional"] # Set to true to turn the models bi-directional

# Create dir is it doesn't exist yet
path = snakemake.output[0]
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs("{}/tables".format(path))
    os.makedirs("{}/figures".format(path))
    os.makedirs("{}/models".format(path))

# Set up data
data = np.load(snakemake.input[0])
labels = np.load(snakemake.input[1])
train_set_size = int(np.ceil(data.shape[0] * 0.9))

# Swap axes to have (Set_size x Seq_len x Features) size
train_x = data[:train_set_size]
train_x = np.swapaxes(train_x, -1, 1)
train_y = labels[:train_set_size]

valid_x = data[train_set_size:]
valid_x = np.swapaxes(valid_x, -1, 1)
valid_y = labels[train_set_size:]

# Set up metrics storage
train_metrics_means = np.zeros((EPOCHS, 4))
valid_metrics_means = np.zeros((EPOCHS, 4))
train_losses_means = np.zeros((EPOCHS))
fpr_means = np.zeros((3, 1311))
tpr_means = np.zeros((3, 1311))

train_metrics_errors = np.zeros((EPOCHS, 4))
valid_metrics_errors = np.zeros((EPOCHS, 4))
train_losses_errors = np.zeros((EPOCHS))
valid_losses_errors = np.zeros((EPOCHS))
fpr_errors = np.zeros((3, 1311))
tpr_errors = np.zeros((3, 1311))

# Run experiment NUMBER_OF_RUNS times
for i in range(NUMBER_OF_RUNS):
	current_experiment = "{}-{}".format(EXPERIMENT_NAME, i+1)
	print("\n\nTraining model no {}\n\n".format(i+1))

	# If saves are present, skip iterations until there is no next save and load the saved metrics
	if os.path.exists("{}/models/{}.pt".format(path, current_experiment)):
		if not os.path.exists("{}/models/{}-{}.pt".format(path, EXPERIMENT_NAME, i+2)):
			train_metrics_means = np.load("{}/tables/train_metrics_means.npy".format(path), allow_pickle=True)
			valid_metrics_means = np.load("{}/tables/valid_metrics_means.npy".format(path), allow_pickle=True)
			train_losses_means = np.load("{}/tables/train_losses_means.npy".format(path), allow_pickle=True)
			valid_losses_means = np.load("{}/tables/valid_losses_means.npy".format(path), allow_pickle=True)
			fpr_means = np.load("{}/tables/fpr_means.npy".format(path), allow_pickle=True)
			tpr_means = np.load("{}/tables/tpr_means.npy".format(path), allow_pickle=True)

			train_metrics_errors = np.load("{}/tables/train_metrics_errors.npy".format(path), allow_pickle=True)
			valid_metrics_errors = np.load("{}/tables/valid_metrics_errors.npy".format(path), allow_pickle=True)
			train_losses_errors = np.load("{}/tables/train_losses_errors.npy".format(path), allow_pickle=True)
			valid_losses_errors = np.load("{}/tables/valid_losses_errors.npy".format(path), allow_pickle=True)
			fpr_errors = np.load("{}/tables/fpr_errors.npy".format(path), allow_pickle=True)
			tpr_errors = np.load("{}/tables/tpr_errors.npy".format(path), allow_pickle=True)

		continue

	# Train a model and return all metrics
	train_metrics, valid_metrics, train_losses, valid_losses, fpr, tpr = train_model(current_experiment, MODEL_TYPE, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS,
																					 LAYERS, DROPOUT, BIDIRECTIONAL, train_x, train_y, valid_x, valid_y, path)
	# Incrementally calculate means and variances
	if i == 0:
		train_metrics_means = train_metrics
		valid_metrics_means = valid_metrics
		train_losses_means = train_losses
		valid_losses_means = valid_losses
		fpr_means = fpr
		tpr_means = tpr

	else:
		# Remember the previous values
		old_train_metrics_means = train_metrics_means
		old_valid_metrics_means = valid_metrics_means
		old_train_losses_means = train_losses_means
		old_valid_losses_means = valid_losses_means
		old_fpr_means = fpr_means
		old_tpr_means = tpr_means
		
		# Update means
		train_metrics_means += (train_metrics - old_train_metrics_means)/(i+1)
		valid_metrics_means += (valid_metrics - old_valid_metrics_means)/(i+1) 
		train_losses_means += (train_losses - old_train_losses_means)/(i+1)
		valid_losses_means += (valid_losses - old_valid_losses_means)/(i+1)
		fpr_means += (fpr - fpr_means)/(i+1)
		tpr_means += (tpr - tpr_means)/(i+1)
		
		# Update variances
		train_metrics_errors += (train_metrics - old_train_metrics_means)*(train_metrics - train_metrics_means)
		valid_metrics_errors += (valid_metrics - old_valid_metrics_means)*(valid_metrics - valid_metrics_means)
		train_losses_errors += (train_losses - old_train_losses_means)*(train_losses - train_losses_means)
		valid_losses_errors += (valid_losses - old_valid_losses_means)*(valid_losses - valid_losses_means)
		fpr_errors += (fpr - old_fpr_means)*(fpr - fpr_means)
		tpr_errors += (tpr - old_tpr_means)*(tpr - tpr_means)

	# Save progress for restarts
	np.save("{}/tables/train_metrics_means.npy".format(path), train_metrics_means)
	np.save("{}/tables/valid_metrics_means.npy".format(path), valid_metrics_means)
	np.save("{}/tables/train_losses_means.npy".format(path), train_losses_means)
	np.save("{}/tables/valid_losses_means.npy".format(path), valid_losses_means)
	np.save("{}/tables/fpr_means.npy".format(path), fpr_means)
	np.save("{}/tables/tpr_means.npy".format(path), tpr_means)

	np.save("{}/tables/train_metrics_errors.npy".format(path), train_metrics_errors)
	np.save("{}/tables/valid_metrics_errors.npy".format(path), valid_metrics_errors)
	np.save("{}/tables/train_losses_errors.npy".format(path), train_losses_errors)
	np.save("{}/tables/valid_losses_errors.npy".format(path), valid_losses_errors)
	np.save("{}/tables/fpr_errors.npy".format(path), fpr_errors)
	np.save("{}/tables/tpr_errors.npy".format(path), tpr_errors)


# Now the errors variables contain variances, transform to standard error of the mean
train_metrics_errors = np.sqrt(train_metrics_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)
valid_metrics_errors = np.sqrt(valid_metrics_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)
train_losses_errors = np.sqrt(train_losses_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)
valid_losses_errors = np.sqrt(valid_losses_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)
fpr_errors = np.sqrt(fpr_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)
tpr_errors = np.sqrt(tpr_errors / (NUMBER_OF_RUNS-1)) / np.sqrt(NUMBER_OF_RUNS)


# Record mean and error calculation results
results_file = open("{}/tables/results.txt".format(path), "w")
for i in range(EPOCHS):
	results_file.write("Epoch No:\t{}\n".format(i+1))
	results_file.write("x\tAccuracy\tPrecision\tRecall\t\tF1 Score\tLoss\n")
	results_file.write("Train\t{:.2f}+-{:.2f}\t\t".format(train_metrics_means[i,0], train_metrics_errors[i,0]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(train_metrics_means[i,1], train_metrics_errors[i,1]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(train_metrics_means[i,2], train_metrics_errors[i,2]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(train_metrics_means[i,3], train_metrics_errors[i,3]))
	results_file.write("{:.5f}+-{:.5f}\n".format(train_losses_means[i], train_losses_errors[i]))
	
	results_file.write("Valid\t{:.2f}+-{:.2f}\t\t".format(valid_metrics_means[i,0], valid_metrics_errors[i,0]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(valid_metrics_means[i,1], valid_metrics_errors[i,1]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(valid_metrics_means[i,2], valid_metrics_errors[i,2]))
	results_file.write("{:.2f}+-{:.2f}\t\t".format(valid_metrics_means[i,3], valid_metrics_errors[i,3]))
	results_file.write("{:.5f}+-{:.5f}\n".format(valid_losses_means[i], valid_losses_errors[i]))

	results_file.write("\n")

x_axis = np.linspace(1, EPOCHS, num=EPOCHS)

# Plot train metrics
plt.errorbar(x_axis, train_metrics_means[:,0], yerr=train_metrics_errors[:,0], label="Accuracy")
plt.errorbar(x_axis, train_metrics_means[:,1], yerr=train_metrics_errors[:,1], label="Precision")
plt.errorbar(x_axis, train_metrics_means[:,2], yerr=train_metrics_errors[:,2], label="Recall")
plt.errorbar(x_axis, train_metrics_means[:,3], yerr=train_metrics_errors[:,3], label="F1 Score")
plt.title("Training Metrics")
plt.xlabel("Epoch Number")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid("on")
plt.savefig("{}/figures/train_metrics.pdf".format(path))
plt.clf()

# Plot validation metrics
plt.errorbar(x_axis, valid_metrics_means[:,0], yerr=valid_metrics_errors[:,0], label="Accuracy")
plt.errorbar(x_axis, valid_metrics_means[:,1], yerr=valid_metrics_errors[:,1], label="Precision")
plt.errorbar(x_axis, valid_metrics_means[:,2], yerr=valid_metrics_errors[:,2], label="Recall")
plt.errorbar(x_axis, valid_metrics_means[:,3], yerr=valid_metrics_errors[:,3], label="F1 Score")
plt.title("Validation Metrics")
plt.xlabel("Epoch Number")
plt.ylabel("%")
plt.legend(loc="upper left")
plt.grid("on")
plt.savefig("{}/figures/valid_metrics.pdf".format(path))
plt.clf()

# Plot losses
plt.errorbar(x_axis, train_losses_means, yerr=train_losses_errors, label="train_loss")
plt.errorbar(x_axis, valid_losses_means, yerr=valid_losses_errors, label="valid_loss")
plt.title("Losses")
plt.xlabel("Epoch Number")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.grid("on")
plt.savefig("{}/figures/losses.pdf".format(path))
plt.clf()

# Plot ROC curves
plt.errorbar(fpr_means[0], tpr_means[0], xerr=fpr_errors[0], yerr=tpr_errors[0], ls='--', label="Germline Variant")
plt.errorbar(fpr_means[1], tpr_means[1], xerr=fpr_errors[1], yerr=tpr_errors[1], ls='--', label="Somatic Variant")
plt.errorbar(fpr_means[2], tpr_means[2], xerr=fpr_errors[2], yerr=tpr_errors[2], ls='--', label="Normal")
x = np.linspace(0, 1, 2)
plt.plot(x)
plt.title("Validation ROC")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="best")
plt.grid('on')
plt.savefig("{}/figures/valid_roc.pdf".format(path))