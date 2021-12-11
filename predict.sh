#!/bin/bash
#
#SBATCH --mail-user=spencerng@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --output=/home/spencerng/slurm/predict-squad.stdout
#SBATCH --error=/home/spencerng/slurm/predict-squad.stderr
#SBATCH --chdir=/home/spencerng/squad-sentiment
#SBATCH --job-name=predict_squad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=32000MB
#SBATCH --partition=titan
#SBATCH --time=4:00:00


python3 predict.py --prefix 0.2 --modify_question
python3 predict.py --prefix 0.3 --modify_question
python3 predict.py --prefix 0.4 --modify_question
python3 predict.py --prefix 0.5 --modify_question
python3 predict.py --prefix 0.6 --modify_question
python3 predict.py --prefix 0.2 --modify_context
python3 predict.py --prefix 0.3 --modify_context
python3 predict.py --prefix 0.4 --modify_context
python3 predict.py --prefix 0.5 --modify_context
python3 predict.py --prefix 0.6 --modify_context
python3 predict.py --prefix 0.2 --modify_context --modify_question
python3 predict.py --prefix 0.3 --modify_context --modify_question
python3 predict.py --prefix 0.4 --modify_context --modify_question
python3 predict.py --prefix 0.5 --modify_context --modify_question
python3 predict.py --prefix 0.6 --modify_context --modify_question