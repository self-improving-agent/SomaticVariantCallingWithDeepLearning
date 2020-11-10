#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N project-somatic-variant-calling-with-deep-learning
#$ -cwd
#$ -pe sharedmem 2
#$ -l h_rt=48:00:00
#$ -M s1618820@sms.ed.ac.uk
#$ -m beas
#$ -l h_vmem=2G

source activate project-somatic-variant-calling-with-deep-learning

snakemake --cores 2 -s pipelines/Snakefile

source deactivate
