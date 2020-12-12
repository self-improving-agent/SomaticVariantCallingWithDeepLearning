#!/bin/sh
# Grid Engine options (lines prefixed with #$)              
#$ -cwd                  
#$ -l h_rt=1:00:00 
#$ -l h_vmem=5G
#$ -M s1618820@sms.ed.ac.uk
#$ -m abe
#$ -N train_model

. /etc/profile.d/modules.sh

module load anaconda/5.0.1

source activate somatic-variant-calling

# number_of_runs, experiment_name, model_type - GRU, LSTM, RNN, Transformer, Perceptron, default is GRU, epochs, batch_size, learning_rate, hidden_units, hidden_layers, dropout, bidirectional

snakemake run_experiment --config number_of_runs=10 experiment_name="GRU" model_type="GRU" epochs=1000 batch_size=250 learning_rate=0.001 hidden_units=32 hidden_layers=2 dropout=0.0 bidirectional=False

source deactivate
