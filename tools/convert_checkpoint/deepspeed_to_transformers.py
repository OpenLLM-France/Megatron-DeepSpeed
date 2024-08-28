#!/usr/bin/env python

import os
import torch
import json

from deepspeed_checkpoint import DeepSpeedCheckpoint
from deepspeed_to_megatron import _create_rank_checkpoint, parse_arguments

# the import was tested to work with this version
# https://github.com/huggingface/transformers/commit/0af901e83 if it diverges we may consider
# copying that version here instead
from convert_megatron_llama2_checkpoint import convert_megatron_checkpoint, recursive_print
from transformers import LlamaConfig

def main():

    # this first part comes mainly from deepspeed_to_megatron.main
    args = parse_arguments()
    print(f'Converting DeepSpeed checkpoint in {args.input_folder} to HF Transformers checkpoint in {args.output_folder}')

    ds_checkpoint = DeepSpeedCheckpoint(args.input_folder, args.target_tp, args.target_pp)
    iteration = ds_checkpoint.get_iteration()
    input_state_dict = _create_rank_checkpoint(ds_checkpoint, 0, 0, args.for_release)

    megatron_args = input_state_dict['args']

    # the 2nd part comes from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint.main
    # Spell out all parameters in case the defaults change.
    config = LlamaConfig(
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=0,
        eos_token_id=1,
        hidden_act='silu',
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        max_position_embeddings=megatron_args.max_position_embeddings,
        model_type='llama',
        num_attention_heads=megatron_args.num_attention_heads,
        num_hidden_layers=megatron_args.num_layers,
        num_key_value_heads=megatron_args.num_key_value_heads,
        rms_norm_eps=megatron_args.layernorm_epsilon,
        rope_theta=500000,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=65024,
        torch_dtype='float16',
    )

    # recursive_print(None, input_state_dict)

    # Convert.
    print("Converting to HF Checkpoint")
    output_state_dict = convert_megatron_checkpoint(input_state_dict, config)

    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Print the structure of converted state dict.
    # recursive_print(None, output_state_dict)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["LlamaForCausalLM"]
    output_config["model_type"] = "llama"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)

    print("Now add tokenizer files and upload to the hub")

if __name__ == "__main__":
    main()
