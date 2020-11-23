#!/bin/sh
# Grid Engine options (lines prefixed with #$)              
#$ -cwd                  
#$ -l h_rt=1:00:00 
#$ -l h_vmem=5G
#$ -M s1618820@sms.ed.ac.uk
#$ -m abe
#$ -N test_model

. /etc/profile.d/modules.sh

module load anaconda/5.0.1

source activate somatic-variant-calling

# experiment_name, test_name, model_type - GRU, LSTM, RNN, Perceptron, default is GRU, hidden_units, hidden_layers, dropout

snakemake test_model --config experiment_name="GRU" test_name='test1' model_type="GRU" hidden_units=32 hidden_layers=2 dropout=0.0 bidirectional=False

source deactivate
