#!/bin/bash
#SBATCH --ntasks=1        
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread   
#SBATCH --account=qgz@cpu      
#SBATCH --qos=qos_cpu-dev      
#SBATCH --time 1:00:00              
#SBATCH --output=%x-%j.out           
#SBATCH --job-name=convert

module purge
module load cpuarch/amd 
module load anaconda-py3/2023.09 
module load cuda/12.1.0 
module load gcc/12.2.0
conda activate lucie

cd ~/Lucie-Training/Megatron-DeepSpeed/

input_folder=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step25000
output_folder=/lustre/fsn1/projects/rech/qgz/commun/transformer_checkpoints/global_step25000

python tools/convert_checkpoint/deepspeed_to_transformers.py --input_folder $input_folder --output_folder $output_folder 

# python tools/convert_checkpoint/inspect_deepspeed_checkpoint.py --folder $input_folder 

