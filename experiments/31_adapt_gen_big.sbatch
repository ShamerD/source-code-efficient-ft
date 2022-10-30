#!/bin/bash
#SBATCH --job-name="31-a-gen"
#SBATCH --time=1-12:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="31-%a.out"
#SBATCH --error="31-%a.err"
#SBATCH --array=32,64,128,256,512

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
    --task concode \
    --sub_task none \
    --use_defaults \
    --batch_size 24 \
    --res_fn results/codet5_base_concode_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size $idx \
    --seed 1

