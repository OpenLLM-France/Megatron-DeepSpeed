import argparse
import os
import re
import zipfile

import torch

from transformers import AutoTokenizer, LlamaConfig


####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


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


####################################################################################################


def convert_megatron_checkpoint(input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # The number of heads.
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads 
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    lm = model["language_model"]

    # Word embeddings.
    output_state_dict["model.embed_tokens.weight"] = lm["embedding"]["word_embeddings"]["weight"][: config.vocab_size, :]

    # The transformer.
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            print(f"Not a layer: {key}")
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)
        assert weight_or_bias == 'weight' # No bias in Llama2

        # The name of the layer.
        layer_name = f"model.layers.{layer_idx}"

        # For layernorm(s), simply store the layer norm.
        if op_name.endswith("layernorm"):
            ln_name = "input_layernorm" if op_name.startswith("input") else "post_attention_layernorm"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # QKV matrix - no GQA 
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ):
            val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Store.
            q_proj, k_proj, v_proj = val.chunk(3, dim=0)
            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q_proj
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_proj
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v_proj
        # QKV matrix - GQA 
        elif (
            op_name == "self_attention.query"
        ):
            q_proj = fix_query_key_value_ordering(val, checkpoint_version, 1, heads, hidden_size_per_head)
            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = q_proj

        elif (
            op_name == "self_attention.key_value"
        ):
            val = fix_query_key_value_ordering(val, checkpoint_version, 2, kv_heads, hidden_size_per_head)
            k_proj, v_proj = val.chunk(2, dim=0)
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_proj
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = v_proj
         
        elif op_name == "self_attention.dense":
            output_state_dict[layer_name + ".self_attn.o_proj.weight"] = val

        elif op_name == "mlp.dense_h_to_4h":
            gate_proj, up_proj = val.chunk(2, dim=0)
            output_state_dict[layer_name + ".mlp.gate_proj.weight"] = gate_proj
            output_state_dict[layer_name + ".mlp.up_proj.weight"] = up_proj

        elif op_name == "mlp.dense_4h_to_h":
            output_state_dict[layer_name + ".mlp.down_proj.weight"] = val

        else:
            raise(NotImplementedError)

        rope_theta = 10000
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_size_per_head, 2).float() / hidden_size_per_head))
        output_state_dict[layer_name + '.self_attn.rotary_emb.inv_freq'] = inv_freq

    assert config.num_hidden_layers == layer_idx + 1

    # The final layernorm
    output_state_dict["model.norm.weight"] = transformer[f"layers.{layer_idx + 1}.weight"]
    output_state_dict["lm_head.weight"] = transformer["final_layernorm.lm_head.weight"]

    return output_state_dict
