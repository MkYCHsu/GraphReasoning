#!/bin/bash
#SBATCH --job-name="mainKG"
###SBATCH --partition=sched_mit_buehler
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:6

###SBATCH -N 8
###SBATCH -n 8

#SBATCH --mem-per-cpu=16G

#SBATCH --time=12:0:0
#SBATCH --output=cout_main.txt
#SBATCH --error=cerr_main.txt
###SBATCH --nodelist=node1230

module purge
source ~/.bashrc
source ~/ml.sh
rm core*
#conda deactivate
conda activate llm

python make_mainKG_TSMC.py


