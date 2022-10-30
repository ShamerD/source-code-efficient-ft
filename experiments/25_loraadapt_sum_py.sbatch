#!/bin/bash
#SBATCH --job-name="25-fa-pysum"
#SBATCH --time=1-06:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="25-%a.out"
#SBATCH --error="25-%a.err"
#SBATCH --array=1,2,4,8,16

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
    --task summarize \
    --sub_task python \
    --use_defaults \
    --batch_size 32 \
    --res_fn results/codet5_base_summarize_python_fflora_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size $idx \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r $idx \
    --seed 1

