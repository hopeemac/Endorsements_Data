#!/bin/bash

#SBATCH --ntasks=3
#SBATCH --time=00:20:00
#SBATCH --partition=parallel
#SBATCH --mem=1000
#SBATCH --account=stking-lab

module load anaconda3/2.4.1
python 'tweets_daily_counts_PAR.py'
