#!/bin/bash
#SBATCH --job-name="175-a-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="175-%a.out"
#SBATCH --error="175-%a.err"
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
    --sub_task cs-java \
    --use_defaults \
    --src_len 256 \
    --trg_len 200 \
    --res_fn results/codet5_base_translate_cj_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size 16 \
    --seed $idx

