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

# experiment_name, model_type - GRU, LSTM, RNN, Perceptron, default is GRU, epochs, batch_size, learning_rate, hidden_units, hidden_layers, dropout, bidirectional

snakemake train_model --config experiment_name="GRU" model_type="GRU" epochs=100 batch_size=25 learning_rate=0.001 hidden_units=32 hidden_layers=2 dropout=0.0 bidirectional=False

source deactivate
