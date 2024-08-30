#!/bin/bash
#SBATCH --job-name="mainKG"
###SBATCH --partition=sched_mit_buehler
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:1

#SBATCH -N 1
#SBATCH -n 8

#SBATCH --mem-per-cpu=32G

#SBATCH --time=12:0:0
#SBATCH --output=cout_main.txt
#SBATCH --error=cerr_main.txt
###SBATCH --nodelist=node1230

module purge
source ~/.bashrc
source ~/ml.sh
rm core*
#conda deactivate
conda activate LLM

python make_mainKG_TSMC.py


