#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N pre_process_test_set
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_rt=1:00:00 
#$ -l h_vmem=1G
#$ -M s1618820@sms.ed.ac.uk
#$ -m beas

. /etc/profile.d/modules.sh

module load anaconda/5.0.1

source activate somatic-variant-calling

snakemake --cores 2 pre_process_test_set --config CHR=20 purity=0.6

source deactivate
