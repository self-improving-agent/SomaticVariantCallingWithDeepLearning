import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from numpy.random import RandomState

from model_training import train_model
from genotyping_model_training import genotyping_train_model
from model_testing import test_model
from genotyping_model_testing import genotyping_test_model

sys.path.append("..")
from visualization.produce_plots import produce_plots
from visualization.genotyping_produce_plots import genotyping_produce_plots
from utils.aggregate_results import aggregate_results
from utils.genotyping_aggregate_results import genotyping_aggregate_results


NUMBER_OF_RUNS = snakemake.config["number_of_runs"]
EXPERIMENT_NAME = snakemake.config["experiment_name"]
MODEL_TYPE = snakemake.config["model_type"]
MODE = snakemake.config["mode"]

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
    if MODE != "genotyping_":
	    os.makedirs("{}/figures/train".format(path))
	    os.makedirs("{}/figures/valid".format(path))
    os.makedirs("{}/figures/ROCs".format(path))
    os.makedirs("{}/models".format(path))

# Set up data
data = np.load(snakemake.input[0])
labels = np.load(snakemake.input[1])
train_set_size = int(np.ceil(data.shape[0] * 0.9))
rand = RandomState(1618820) # Student number as random seed

# Swap axes to have (Set_size x Seq_len x Features) size
train_x = data[:train_set_size]
train_x = np.swapaxes(train_x, -1, 1)
train_y = labels[:train_set_size]

valid_x = data[train_set_size:]
valid_x = np.swapaxes(valid_x, -1, 1)
valid_y = labels[train_set_size:]

# Run experiment NUMBER_OF_RUNS times
for i in range(NUMBER_OF_RUNS):
	current_experiment = "{}-{}".format(EXPERIMENT_NAME, i+1)
	print("\n\nTraining model no {}\n\n".format(i+1))

	# If saves are present, skip iterations until there is no next save and load the saved metrics
	if os.path.exists("{}/models/{}.pt".format(path, current_experiment)):
		continue

	# Shuffle training data
	shuffle_order = rand.permutation(train_x.shape[0])
	train_x = train_x[shuffle_order]
	train_y = train_y[shuffle_order]

	# Train a model and return all metrics
	if MODE != "genotyping_":
		train_model(current_experiment, MODEL_TYPE, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS,
				LAYERS, DROPOUT, BIDIRECTIONAL, train_x, train_y, valid_x, valid_y, path)
	else:
		genotyping_train_model(current_experiment, MODEL_TYPE, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS,
				LAYERS, DROPOUT, BIDIRECTIONAL, train_x, train_y, valid_x, valid_y, path)


# After all models are trained, produce aggregate plots
if MODE != "genotyping_":
	produce_plots(EXPERIMENT_NAME)
else:
	genotyping_produce_plots(EXPERIMENT_NAME)

# Also calculate aggregate results for the final epoch for convieninent performance comparision
if MODE != "genotyping_":
	aggregate_results(EXPERIMENT_NAME)
else:
	genotyping_aggregate_results(EXPERIMENT_NAME)

# TESTING

# Create dir is it doesn't exist yet
if not os.path.exists("{}/tables/test".format(path)):
    os.makedirs("{}/tables/test".format(path))

# Load test data
test_x = np.load(snakemake.input[2])
test_x = np.swapaxes(test_x, -1, 1)
test_y = np.load(snakemake.input[3])


# Evaluate all experiments on test set
print("Begin evaluation")
for i in range(NUMBER_OF_RUNS):
	current_experiment = "{}-{}".format(EXPERIMENT_NAME, i+1)
	print("\n\nTesting model no {}\n\n".format(i+1))

	# If saves are present, skip iterations until there is no next save and load the saved metrics
	if os.path.exists("{}/tables/test/{}-test-metrics.txt".format(path, current_experiment)):
		continue

	# Test a model
	if MODE != "genotyping_":
		test_model(current_experiment, MODEL_TYPE, HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL, 
		       test_x, test_y, path)
	else:
		genotyping_test_model(current_experiment, MODEL_TYPE, HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL, 
		       test_x, test_y, path)

# Calculate aggregate results for the test metrics
if MODE != "genotyping_":
	aggregate_results(EXPERIMENT_NAME, mode="Test")
else:
	genotyping_aggregate_results(EXPERIMENT_NAME, mode="Test")