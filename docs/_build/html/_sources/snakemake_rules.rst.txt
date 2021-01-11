Snakemake Rules
================
This page describes the Snakemake rules used in the project, including all configuration parameters that need to be included in the bash scripts created to call these rules. The similar rules are grouped into categories. For example usages see the bash scripts created during the project, which are saved in the workflows folder.

Pre-processing
----------------------------------
Pre-processing happens with the rules:

* pre_process_train_set
* pre_process_test_set

The train set one processes the 22nd chromosome's genomic region of the Ashkenazi son as a normal sample and of the standard reference genome at unique mutation locations mixed in as the tumor sample. The test set one does the same but with the 20th chromosome. Other than outputs and the bed file input the two rules are identical.

There are 2 configuration parameter for this rule:

* purity - represents the % of reads taken from the tumor sample at its unique mutation locations
* CHR - the chromosome to be processed

This is the first stage of the pipeline, hence the inputs are the corresponding BAM and VCF files, while the outputs are .txt files of the dataset. The script used is pre_processing.py.


Dataset Building
----------------------------------
The next stage of the pipeline is using the pre-processed data to build the dataset consumed by the neural networks trained afterwards. The rules for this are:

* build_train_dataset
* build_test_dataset

With the configuration parameter:

* context_size - how many number of positions should be considered around a target site when building the sequential data (e.g. 5 will yield a window of 5 nucleotides upstream and downstream, resulting in a total sequence length of 2*5+1 = 11)

The input to these rules are the data files to be processed, the outputs are a .npy format dataset and labels file for both training and test sets. The script used is building_dataset.py

Neural Network Training & Testing
----------------------------------
The rule in this category responsible for training neural network models. It trains multiple models (specified by the number_of_runs) and records the average performance and variances to calculate errors.

* run_experiment

This rule has 9 configuration parameters, to allow for greater flexibility in creating the desired network. These are:

* number_of_runs - the number of models to be trained as part of the experiment
* experiment_name - the name of the experiment, this will also be the name of the results folder
* model_type - sets what kind of architecture should the model use. The options are: GRU, LSTM, RNN, Transformer, Perceptron, with the default being GRU
* epochs - for how many epochs should the model train for
* batch_size - how many datapoints should be in one batch
* learning_rate - the learning rate of the model
* hidden_units - number of nodes in the hidden layers
* hidden_layers - number of hidden layers to be used
* dropout - proportion of nodes to be removed when applying dropout layers. If set to 0.0, no dropout is applied
* bidirectional - whether the the model should be only bidirectional (not relevant for Perceptron)

The dataset and labels are taken as input, while the output is a folder named after the experiment_name containing containing 3 sub folders: tables, models and figures. Tables contains the metrics recorded during training for each model and a final aggregated metrics file to show the combined averaged results with errors, along with a sub-folder with a file for the test set results of each model and a similar aggregated results file. Inside models there are the saved models for each run along with the checkpoints saved during the training of each. Figures is filled after the specified number of models are trained (number_of_runs), with 6 graphs monitoring the change in all metrics (Accuracy, Precision, Recall, F1) for the training and validation sets, training and validation losses, and the AUC for the ROC curves on the validation set for each of the 3 classes. Additionally there is a sub-folder with ROC curves for each model trained for both the validation and test sets.

The script used is run_experiment.py which calls train_model from model_training.py for each model to be trained. When finished it calls produce_plots from produce_plots.py in visualization to make all the plots and aggregate_results from aggregate_results.py in utils to create the aggregated metrics file. Then each model is evaluated by calling test_model from model_testing.py and then aggregate_results in test mode for combining the results.

