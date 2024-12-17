import argparse
from transformers import LlamaConfig
import torch
import json
import os
from safetensors.torch import save_file as safe_save_file
from huggingface_hub import split_torch_state_dict_into_shards

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input Universal Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output HF checkpoint folder')
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="10GB",
        help=(
            "The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size "
            "lower than this size. If expressed as a string, needs to be digits followed by a unit (like `5MB`). "
            "Only used when converting a Megatron checkpoint to a Transformers checkpoint."
        ),
    )
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
        pad_token_id=3,
        hidden_act='silu',
        hidden_size=megatron_args.hidden_size,
        intermediate_size=megatron_args.ffn_hidden_size,
        max_position_embeddings=megatron_args.max_position_embeddings,
        model_type='llama',
        num_attention_heads=megatron_args.num_attention_heads,
        num_hidden_layers=megatron_args.num_layers,
        num_key_value_heads=megatron_args.num_key_value_heads,
        rms_norm_eps=megatron_args.layernorm_epsilon,
        rope_theta=megatron_args.rope_theta,
        tie_word_embeddings=False,
        use_cache=True,
        vocab_size=65024,
        torch_dtype='bfloat16',
        transformers_version = "4.36.1",
    )

    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    
    zero_path = os.path.join(args.input_folder, "zero")

    output_state_dict = {}

    # Embedding
    embedding_weight = torch.load(os.path.join(zero_path, "1.word_embeddings.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
    output_state_dict["model.embed_tokens.weight"] = embedding_weight[: config.vocab_size, :]

    # Final Layer
    finalnorm_weight = torch.load(os.path.join(zero_path, f"{num_layers+2}.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
    lm_head_weight = torch.load(os.path.join(zero_path, f"{num_layers+3}.lm_head.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
    output_state_dict["model.norm.weight"] = finalnorm_weight
    output_state_dict["lm_head.weight"] = lm_head_weight

    print("Converting to HF Checkpoint")
    for l in range(num_layers):
        # Layer norm
        input_layernorm = torch.load(os.path.join(zero_path, f"{l+2}.input_layernorm.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        output_state_dict[f"model.layers.{l}.input_layernorm.weight"] = input_layernorm

        post_attention_layernorm = torch.load(os.path.join(zero_path, f"{l+2}.post_attention_layernorm.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        output_state_dict[f"model.layers.{l}.post_attention_layernorm.weight"] = post_attention_layernorm

        # MLP
        dense_4h_to_h = torch.load(os.path.join(zero_path, f"{l+2}.mlp.dense_4h_to_h.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        output_state_dict[f"model.layers.{l}.mlp.down_proj.weight"] = dense_4h_to_h

        dense_h_to_4h = torch.load(os.path.join(zero_path, f"{l+2}.mlp.dense_h_to_4h.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        gate_proj, up_proj = dense_h_to_4h.chunk(2, dim=0)
        output_state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = gate_proj
        output_state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = up_proj

        # Attention
        query = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.query.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        q_proj = fix_query_key_value_ordering(query, checkpoint_version, 1, num_attention_heads, hidden_size_per_head)
        output_state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = q_proj

        key_value = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.key_value.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        key_value = fix_query_key_value_ordering(key_value, checkpoint_version, 2, num_key_value_heads, hidden_size_per_head)
        k_proj, v_proj = key_value.chunk(2, dim=0)
        output_state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = k_proj
        output_state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = v_proj
                
        dense = torch.load(os.path.join(zero_path, f"{l+2}.self_attention.dense.weight/fp32.pt"), map_location=torch.device('cpu'))['param'].bfloat16()
        output_state_dict[f"model.layers.{l}.self_attn.o_proj.weight"] = dense

        # Rotary
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, hidden_size_per_head, 2).float() / hidden_size_per_head))
        output_state_dict[f"model.layers.{l}.self_attn.rotary_emb.inv_freq"] = inv_freq

    os.makedirs(args.output_folder, exist_ok=True)

    # Store the config to file.
    output_config_file = os.path.join(args.output_folder, "config.json")
    output_config = config.to_dict()
    output_config["architectures"] = ["LlamaForCausalLM"]
    output_config["model_type"] = "llama"
    print(f'Saving config to "{output_config_file}"')
    with open(output_config_file, "w") as f:
        json.dump(output_config, f, indent=2)

    def save_state_dict(state_dict, save_directory):
        state_dict_split = split_torch_state_dict_into_shards(state_dict)
        for filename, tensors in state_dict_split.filename_to_tensors.items():
            shard = {tensor: state_dict[tensor] for tensor in tensors}
            safe_save_file(
                shard,
                os.path.join(save_directory, filename),
                metadata={"format": "pt"},
            )
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
                f.write(json.dumps(index, indent=2))

    save_state_dict(output_state_dict, args.output_folder)

    print("Now add tokenizer files and upload to the hub")

if __name__ == "__main__":
    main()
