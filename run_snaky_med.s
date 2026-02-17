#!/bin/bash

#SBATCH --job-name=snaky
#SBATCH --output=SLURM/snaky_%j.out
#SBATCH --error=SLURM/snaky_%j.err
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH -p public-cpu
#SBATCH --mem=10G
#SBATCH --exclude=cpu012,cpu014

# run the simulation

python snaky_trigger.py -s $1 -i $2 -b $3 -e $4 -H 0 -A 0 -S matching_diff
echo 'SUCCESS COMPUTING SNAKY'



