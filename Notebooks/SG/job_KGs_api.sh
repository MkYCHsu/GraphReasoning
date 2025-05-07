#!/bin/bash
#SBATCH --job-name="KGs"
###SBATCH --partition=sched_mit_buehler
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:4

#SBATCH -N 1
#SBATCH -n 8

#SBATCH --mem-per-cpu=4G

#SBATCH --time=12:0:0
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt
###SBATCH --nodelist=node1230

module purge
source ~/.bashrc
source ~/ml.sh
rm core*
#conda deactivate
conda activate llm

#python make_KGs_SG_api.py
python make_KGs_SG_abstract_api.py



