#!/bin/bash

source ~/Lucie-Training/training/set_env.sh
cd ~/Lucie-Training/Megatron-DeepSpeed/
export PYTHONPATH=~/Lucie-Training/Megatron-DeepSpeed


##### Method 1
# DS to Transformer
input_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000
output_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000_transformer_bis

python tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder $input_folder --output_folder $output_folder 


##### Method 2
# DS to Universal
input_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000
output_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000_universal

python tools/convert_checkpoint/ds_to_universal.py --input_folder $input_folder --output_folder $output_folder 

# Universal to Transformer
input_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000_universal
output_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000_transformer

python tools/convert_checkpoint/universal_to_hf_llama.py --input_folder $input_folder --output_folder $output_folder 


