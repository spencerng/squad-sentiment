#!/bin/bash
#
#SBATCH --mail-user=spencerng@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/spencerng/slurm/squad.stdout
#SBATCH --error=/home/spencerng/slurm/squad.stderr
#SBATCH --chdir=/home/spencerng/squad-sentiment
#SBATCH --job-name=train_squad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=12000MB
#SBATCH --partition=titan
#SBATCH --time=4:00:00


python3 train.py