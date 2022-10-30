#!/bin/bash
#SBATCH --job-name="40-c-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="40-%a.out"
#SBATCH --error="40-%a.err"
#SBATCH --array=2,3

module purge
module load Python/Anaconda_v11.2021

source deactivate
source activate codelora

idx=$SLURM_ARRAY_TASK_ID

#Executable
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_CACHE=~/.cache/huggingface/transformers/
python ~/code-lora/CodeT5/sh/run.py \
    --model_tag codet5_base \
    --task translate \
    --sub_task java-cs \
    --use_defaults \
    --src_len 256 \
    --trg_len 200 \
    --res_fn results/codet5_base_translate-jc.txt \
    --seed $idx

