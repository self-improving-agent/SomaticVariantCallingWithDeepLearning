import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from model_training import train_model
from genotyping_model_training import genotyping_train_model
from model_testing import test_model
from genotyping_model_testing import genotyping_test_model

sys.path.append("..")
from visualization.produce_plots import produce_plots
from visualization.genotyping_produce_plots import genotyping_produce_plots
from utils.aggregate_results import aggregate_results
from utils.genotyping_aggregate_results import genotyping_aggregate_results


NUMBER_OF_RUNS = 10
EXPERIMENT_NAME = "Transformer_Long"
MODEL_TYPE = "Transformer"
MODE = ""
SAVE_FREQ = 50

# Hyperparameters
EPOCHS = 2000
BATCH_SIZE = 250
LEARNING_RATE = 0.001
HIDDEN_UNITS = 32
LAYERS = 1
DROPOUT = 0.10 # Set to positive to include dropout layers
BIDIRECTIONAL = True # Set to true to turn the models bi-directional

# Create dir is it doesn't exist yet
path = "../../results/{}".format(EXPERIMENT_NAME)
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs("{}/tables".format(path))
    os.makedirs("{}/figures".format(path))
    if MODE != "genotyping_":
	    os.makedirs("{}/figures/train".format(path))
	    os.makedirs("{}/figures/valid".format(path))
	    os.makedirs("{}/figures/test".format(path))
    os.makedirs("{}/figures/ROCs".format(path))
    os.makedirs("{}/models".format(path))
    os.makedirs("{}/models/checkpoints".format(path))
    for i in range(NUMBER_OF_RUNS):
    	os.makedirs("{}/models/checkpoints/{}".format(path, i+1))

# Set up data
data = np.load("../../data/processed/{}train_dataset.npy".format(MODE))
labels = np.load("../../data/processed/{}train_labels.npy".format(MODE))
train_set_size = int(np.ceil(data.shape[0] * 0.9))

# Swap axes to have (Set_size x Seq_len x Features) size
train_x = data[:train_set_size]
train_x = np.swapaxes(train_x, -1, 1)
train_y = labels[:train_set_size]

valid_x = data[train_set_size:]
valid_x = np.swapaxes(valid_x, -1, 1)
valid_y = labels[train_set_size:]

# Run experiment NUMBER_OF_RUNS times
print("Begin experiment")
for i in range(NUMBER_OF_RUNS):
	current_experiment = "{}-{}".format(EXPERIMENT_NAME, i+1)
	print("\n\nTraining model no {}\n\n".format(i+1))

	# If saves are present, skip iterations until there is no next save and load the saved metrics
	if os.path.exists("{}/models/{}.pt".format(path, current_experiment)):
		continue

	# Train a model
	if MODE != "genotyping_":
		train_model(current_experiment, MODEL_TYPE, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS,
				LAYERS, DROPOUT, BIDIRECTIONAL, train_x, train_y, valid_x, valid_y, path, SAVE_FREQ)
	else:
		genotyping_train_model(current_experiment, MODEL_TYPE, EPOCHS, LEARNING_RATE, BATCH_SIZE, HIDDEN_UNITS,
				LAYERS, DROPOUT, BIDIRECTIONAL, train_x, train_y, valid_x, valid_y, path, SAVE_FREQ)

# After all models are trained, produce aggregate plots
# print("Producing plots\n")
# if MODE != "genotyping_":
# 	produce_plots(EXPERIMENT_NAME)
# else:
# 	genotyping_produce_plots(EXPERIMENT_NAME)

# # Also calculate aggregate results for the final epoch for convieninent performance comparision
# print("Aggregating results\n")
# if MODE != "genotyping_":
# 	aggregate_results(EXPERIMENT_NAME)
# else:
# 	genotyping_aggregate_results(EXPERIMENT_NAME)

# TESTING

# Create dir is it doesn't exist yet
if not os.path.exists("{}/tables/test".format(path)):
    os.makedirs("{}/tables/test".format(path))

# Load test data
test_x = np.load("../../data/processed/{}test_dataset.npy".format(MODE))
test_x = np.swapaxes(test_x, -1, 1)
test_y = np.load("../../data/processed/{}test_labels.npy".format(MODE))


# Evaluate all experiments on test set
print("Begin evaluation")
for i in range(NUMBER_OF_RUNS):
	current_experiment = "{}-{}".format(EXPERIMENT_NAME, i+1)
	print("\n\nTesting model no {}\n\n".format(i+1))

	# If saves are present, skip iterations until there is no next save and load the saved metrics
	# if os.path.exists("{}/tables/test/{}-test-metrics.txt".format(path, current_experiment)):
	# 	continue

	# Test a model
	if MODE != "genotyping_":
		test_model(current_experiment, MODEL_TYPE, HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL, 
		       test_x, test_y, path)
	else:
		genotyping_test_model(current_experiment, MODEL_TYPE, HIDDEN_UNITS, LAYERS, DROPOUT, BIDIRECTIONAL, 
		       test_x, test_y, path)

# Produce plots for testing
print("Producing test plots\n")
if MODE != "genotyping_":
	produce_plots(EXPERIMENT_NAME, mode="Test")
else:
	genotyping_produce_plots(EXPERIMENT_NAME, mode="Test")

# Calculate aggregate results for the test metrics
print("Aggregating test tresults\n")
if MODE != "genotyping_":
	aggregate_results(EXPERIMENT_NAME, mode="Test")
else:
	genotyping_aggregate_results(EXPERIMENT_NAME, mode="Test")

print("Experiment concluded!")