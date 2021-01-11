Scripts
=========
This page describes the various Python scripts used throughout the project. For more details see the commented scripts themselves.

Pre-processing
------------------------
The data pre-processing script takes two VCF and two BAM files corresponding to the normal and tumor samples, as well as a bed file detailing the exome locations within the 22nd human chromosome, as input and produces two txt files containing the extracted data from the 22nd chromosome's genomic region of the normal sample with the second file containing some mixed in data from the tumor sample at the mutation locations unique to the tumor sample. These are declared as somatic variants and  a purity configuration parameter determines the percentage of reads to be mixed in. These files are saved in the interim data folder. The libraries used to achieve this are pysam and pyvcf.

The 3 types of labels assigned to locations are: "Normal", "Germline Variant", "Somatic Variant". 

Dataset Building
------------------------
The dataset building script is loading in the txt file produced by pre-processing and builds a labelled dataset, outputting the dataset and the labels separately into the processed data folder. This is done via numpy and the outputs are of the npy format. The context_window cofiguration parameter defines the shape of the data.

One-hot encoding is used for the 3 classes. Datapoints corresponsing to the "Normal" class are randomly sampled, alltogether as many as the sum of the number of datapoints in the other 2 classes.


Run Experiment
------------------------
This is the main script for analysis that wraps model training, testing, plot making and result aggregating in itself. The .npy dataset files are loaded in along with the hyperparameters passed in the configurations, and the outputs are the metrics files and graphs produced during and after the training and testing process, along with the trained models, all saved in a new folder in results. A specified number of models are trained as part of an experiment calling the model training script, then the results are used to create plots via produce plots script and results combined via the aggregate results script. For the evaluation model testing is called for each model and then aggregate results to create a single set of metrics. Plotting is done via seaborn, the plots saved are a pair of curves showing how the given metric evolved over training on both training and validation sets (Accuracy, Precision, Recall, F1, Loss), and a separate graph for the AUCs of the ROC curves for each of the 3 classes on the validation and test set of each model.


Model Training
------------------------
Neural network training happens using the pytorch library, models are choosen from pre-defined types with the passed hyperparameters from run experiment. The Adam optimizer and Cross Entropy loss is used. For metric calculation the confusion matrix utility of sklearn.metrics is used. If a cuda compatible GPU is available, it is utilized for speedier execution.


Model Testing
------------------------
Testing takes a trained model and a test dataset with labels and evaluates the model. The libraries used are the same as for training, the hyperparameters are passed similarly.


Produce Plots
------------------------
Given an experiment, this script makes the graphs displaying metric averages and errors over the course of the training using pandas and seaborn. It uses the produced metrics.txt files and reads all data from them.


Aggregate Results
------------------------
This script takes an experiment and a mode to specify whether the current invocation want to get training and validation set (Train) or test set (Test) results. The corresponding metrics.txt files are read for the relevant data across all epochs of all runs and a single statistic with mean and error is produced for each metric. The results are saved in a new file.

