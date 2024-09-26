#!/bin/bash

export TORCH_CUDA_ARCH_LIST="9.0+PTX" 

source ~/Lucie-Training/training/set_env.sh

CHECKPOINT_PATH=/lustre/fsn1/projects/rech/qgz/commun/checkpoints/pretraining/global_step135000_universal
LOGS_PATH=/lustre/fsn1/projects/rech/qgz/commun/lucie-logs/lucie-logs/pretraining
TOKENIZER_PATH=/lustre/fsn1/projects/rech/qgz/commun/preprocessed_data/Lucie/lucie_tokens_65k_grouped/tokenizer

export CUDA_DEVICE_MAX_CONNECTIONS=1

# ----- model
HIDDEN_SIZE=4096 
FFN_HIDDEN_SIZE=12288 
NUM_LAYERS=32 
NUM_HEADS=32
SEQ_LENGTH=4096 
NUM_KV_HEADS=8 


MICRO_BATCH_SIZE=1
TP=1
PP=1
ZERO_STAGE=0
config_json="$LOGS_PATH/ds_config.json"


DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

GPT_ARGS=" \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --bf16 \
       --no-query-key-layer-scaling \
       "

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export PYTHONPATH=~/Lucie-Training/Megatron-DeepSpeed

cd ~/Lucie-Training/Megatron-DeepSpeed

PROGRAM_CMD="tools/generate_samples_gpt.py  \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path $TOKENIZER_PATH \
       --use-flash-attn-v2 \
       --log-interval 1 \
       --out-seq-length 4096 \
       --temperature 1.0 \
       --genfile unconditional_samples.json \
       --top_p 0.9 \
       --num-samples 0 \
       --load $CHECKPOINT_PATH \
       $GPT_ARGS \
       $DEEPSPEED_ARGS \
       --inference \
       --universal-checkpoint \
       "

GPUS_PER_NODE=4


LAUNCHER_CMD="deepspeed --num_gpus $GPUS_PER_NODE --num_nodes $SLURM_NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

$LAUNCHER_CMD $PROGRAM_CMD

