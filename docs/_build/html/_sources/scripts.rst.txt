Scripts
=========
This page describes the various Python scripts used throughout the project. For more details see the commented scripts themselves.

Pre-processing
------------------------
The data pre-processing script takes two VCF and two BAM files corresponding to the normal and tumor samples, as well as a bed file detailing the exome locations within the 22nd human chromosome, as input and produces two txt files containing the extracted data from the 22nd chromosome's genomic region of the normal sample with the second file containing some mixed in data from the tumor sample at the mutation locations unique to the tumor sample. These are declared as somatic variants and  a purity configuration parameter determines the percentage of reads to be mixed in. These files are saved in the interim data folder. The libraries used to achieve this are pysam and pyvcf.

The 3 types of labels assigned to locations are: "Normal", "Germline Variant", "Somatic Variant". 

Dataset Building
------------------
The dataset building script is loading in the txt file produced by pre-processing and builds a labelled dataset, outputting the dataset and the labels separately into the processed data folder. This is done via numpy and the outputs are of the npy format. The context_window cofiguration parameter defines the shape of the data.

One-hot encoding is used for the 3 classes. Datapoints corresponsing to the "Normal" class are randomly sampled, alltogether as many as the sum of the number of datapoints in the other 2 classes.


Model Training
------------------------
Neural network training happens using the pytorch library. The npy files are loaded in and the outputs are the metrics and graphs produced during and after the training process, along with the trained model itself, all saved in a new folder in results. The graphs saved are: training metrics, validation metrics, losses and validation ROC, all produced from the averaged results with errors obtained from multiple runs. For metric calculation sklearn.metrics is used, for plotting matplotlib.pyplot. All hyperparameters of the model to be trained, in addition to the experiment name and model type are given as configuration parameters, as well as the number of models to be trained for an experiment.


Model Testing
-----------------------
Testing requires as input a trained model and a test dataset with labels. The libraries used are the same as for training. The produced outputs are testing metrics and graphs of output probabilities, class ROCs. The hyperparameters of the model to be tested have to be passed as configuration parametersto initialize the model.


