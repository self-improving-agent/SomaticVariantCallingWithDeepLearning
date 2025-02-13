#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N build_test_dataset
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_rt=1:00:00 
#$ -l h_vmem=8G
#$ -M s1618820@sms.ed.ac.uk
#$ -m beas

. /etc/profile.d/modules.sh

module load anaconda/5.0.1

source activate somatic-variant-calling

# mode - "normal", "genotyping_"

snakemake --cores 2 build_test_dataset --config context_size=40 mode="genotyping_"

source deactivate
