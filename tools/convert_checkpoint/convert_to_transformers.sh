#!/bin/bash
#SBATCH --ntasks=1        
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread   
#SBATCH --account=qgz@cpu      
#SBATCH --qos=qos_cpu-dev      
#SBATCH --time 1:00:00              
#SBATCH --output=%x-%j.out           
#SBATCH --job-name=convert

input_folder=$ALL_CCFRSCRATCH/checkpoints/pretraining/global_step25000
output_folder=$ALL_CCFRSCRATCH/transformer_checkpoints/global_step25000

python deepspeed_to_transformers --input_folder $input_folder --output_folder $output_folder