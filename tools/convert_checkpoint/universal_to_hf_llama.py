import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
import os
import torch
import json
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input Universal Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output HF checkpoint folder')
    args = parser.parse_args()
    print(f'args = {args}')
    return args

def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param

def main():
    args = parse_arguments()

    mp_rank_00_model_states = torch.load(os.path.join(args.input_folder, "mp_rank_00_model_states.pt"))
    megatron_args = mp_rank_00_model_states['args']
    checkpoint_version = mp_rank_00_model_states['checkpoint_version']

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
        torch_dtype='bfloat16',
    )

    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    
    zero_path = os.path.join(args.input_folder, "zero")

    output_state_dict = {}

    # Embedding
    embedding_weight = torch.load(os.path.join(zero_path, "1.word_embeddings.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
    output_state_dict["model.embed_tokens.weight"] = embedding_weight[: config.vocab_size, :]

    # Final Layer
    finalnorm_weight = torch.load(os.path.join(zero_path, f"{num_layers+2}.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
    lm_head_weight = torch.load(os.path.join(zero_path, f"{num_layers+3}.lm_head.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
    output_state_dict["model.norm.weight"] = finalnorm_weight
    output_state_dict["lm_head.weight"] = lm_head_weight

    print("Converting to HF Checkpoint")
    for l in range(num_layers):
        # Layer norm
        input_layernorm = torch.load(os.path.join(zero_path, f"{l+2}.input_layernorm.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        output_state_dict[f"model.layers.{l}.input_layernorm.weight"] = input_layernorm

        post_attention_layernorm = torch.load(os.path.join(zero_path, f"{l+2}.post_attention_layernorm.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        output_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = post_attention_layernorm

        # MLP
        dense_4h_to_h = torch.load(os.path.join(zero_path, f"{l+2}.mlp.dense_4h_to_h.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        output_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = dense_4h_to_h

        dense_h_to_4h = torch.load(os.path.join(zero_path, f"{l+2}.mlp.dense_h_to_4h.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        gate_proj, up_proj = dense_h_to_4h.chunk(2, dim=0)
        output_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = gate_proj
        output_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = up_proj

        # Attention
        query = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.query.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        q_proj = fix_query_key_value_ordering(query, checkpoint_version, 1, num_attention_heads, hidden_size_per_head)
        output_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = q_proj

        key_value = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.key_value.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        key_value = fix_query_key_value_ordering(key_value, checkpoint_version, 2, num_key_value_heads, hidden_size_per_head)
        k_proj, v_proj = key_value.chunk(2, dim=0)
        output_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = k_proj
        output_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = v_proj
                
        dense = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.dense.weight/fp32.pt"), map_location=torch.device('cpu'))['param']
        output_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = dense

        # Rotary
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, hidden_size_per_head, 2).float() / hidden_size_per_head))
        output_state_dict[f"model.layers.{l}.self_attn.rotary_emb.inv_freq"] = inv_freq

    basename = args.output_folder
    os.makedirs(basename, exist_ok=True)

    # Store the config to file.
    output_config_file = os.path.join(basename, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["LlamaForCausalLM"]
    output_config["model_type"] = "llama"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f)

    # Store the state_dict to file.
    # output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    # print(f'Saving checkpoint to "{output_checkpoint_file}"')
    # torch.save(output_state_dict, output_checkpoint_file)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        print(
            f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
            f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
            f"index located at {save_index_file}."
        )

    print("Now add tokenizer files and upload to the hub")

if __name__ == "__main__":
    main()
