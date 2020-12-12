Snakemake Rules
================
This page describes the Snakemake rules used in the project, including all configuration parameters that need to be included in the bash scripts created to call these rules. The similar rules are grouped into categories. For example usages see the bash scripts created during the project, which are saved in the workflows folder.

Pre-processing
------------------------
Pre-processing currenty happens with the rule:

* pre_process

Which processes the 22nd chromosome's genomic region of the Ashkenazi son as a normal sample and of the standard reference genome at unique mutation locations mixed in as the tumor sample.

There is one configuration parameter for this rule:

* purity - represents the % of reads taken from the tumor sample at its unique mutation locations

This is the first stage of the pipeline, hence the inputs are the corresponding BAM and VCF files, while the outputs are .txt files of the dataset. The script used is pre_processing.py.


Dataset Building
-----------------
The next stage of the pipeline is using the pre-processed data to build the dataset consumed by the neural networks trained afterwards. The rule for this is:

* build_dataset

With the configuration parameter:

* context_size - how many number of positions should be considered around a target site when building the sequential data (e.g. 5 will yield a window of 5 nucleotides upstream and downstream, resulting in a total sequence length of 2*5+1 = 11)

The input to this rule is the data to be processed, the outputs are a .npy format dataset and labels file. The script used is building_dataset.py

Neural Network Training
-------------------------
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

The dataset and labels are taken as input, while the output is a folder named after the experiment_name containing containing 3 sub folders: tables, models and figures. Tables contains the metrics recorded during training, for each model as well as the current means and variances (used for the standard errors), and a final metrics file to show the combined averaged results with errors. Inside models there are the saved models for each run along with the checkpoints saved during the training of each. Figures is filled after the specified number of models are trained (number_of_runs), with 4 graphs monitoring the change in training set metrics, validation set metrics, losses and the final validation set ROC. All graphs are produced using the averaged results with errors. The script used is run_experiment.py which calls model_training.py for each model to be trained.

Neural Network Testing
-----------------------
In this final category there is also one rule:

* test_model

This rule has 6 parameters:

* experiment_name - name of the model to be tested
* test_name - name of the test that is being executed, results are saved based on it
* model_type - used to initialize the model to tested, options are same as at training, should be the same value as for the model to be tested (holds for all following parameters)
* hidden_units - used to initialize the model to be tested
* hidden_layers - used to initialize the model to be tested
* dropout - used to initialize the model to be tested
* bidirectional - used to initialize the model to be tested

For these rules the inputs are the selected test dataset and labels, along with the model that will run the inference on these.  The outputs are the metrics of the test (in results.txt), an ROC graph and another graph showcasing the probability outputs of the model for the given test data. The graphs are not created in the case of the last rule. The script used is model_testing.py.

TEST DATASET TO BE DETERMINED!
