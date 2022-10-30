#!/bin/bash
#SBATCH --job-name="151-fa-translate"
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="151-%a.out"
#SBATCH --error="151-%a.err"
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
    --model_tag plbart \
    --task translate \
    --sub_task java-cs \
    --use_defaults \
    --batch_size 32 \
    --lr 3e-5 \
    --warmup 2500 \
    --src_len 256 \
    --trg_len 200 \
    --res_fn results/plbart_translate_jc_fflora_adapter.txt \
    --apply_adapter \
    --adapter_type houlsby \
    --adapter_size 512 \
    --apply_lora_ff \
    --lora_ff_alpha 16 \
    --lora_ff_r 512 \
    --seed $idx

