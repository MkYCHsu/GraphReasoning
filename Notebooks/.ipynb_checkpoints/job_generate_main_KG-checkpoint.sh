#!/bin/bash
#SBATCH --job-name="main_kg"
###SBATCH --partition=sched_mit_buehler
#SBATCH --partition=sched_mit_buehler_gpu
#SBATCH --gres=gpu:1

###SBATCH -N 8
#SBATCH -n 4

#SBATCH --mem-per-cpu=2G

#SBATCH --time=12:0:0
#SBATCH --output=cout.txt
#SBATCH --error=cerr.txt


###SBATCH --nodelist=node[1229-1230]
#SBATCH --exclude=node1228

module purge
source ~/ml.sh
conda deactivate
conda activate LLM

rm *.log
#nvidia-smi
#python -c "import torch; print(torch.__version__)"

python make_mainKG_TSMC.py

#openllm start ~/pool/llm/deepseek-coder-7b-instruct-v1.5

#python3 -m fastchat.serve.cli --model-path ~/pool/CodeLlama-7b-Python-hf #--port 9528

#litellm --model ollama/codellama 
#litellm --model gpt-3.5-turbo
