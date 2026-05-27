def get_cross_attention_inputs_keys(layer_name: str) -> list[str]:
    if layer_name == ".transformer" or layer_name.endswith(".attn2"):
        return ["encoder_hidden_states"]
    elif "attn2" in layer_name and (layer_name.endswith(".to_k") or layer_name.endswith(".to_v")):
        return "all"
    else:
        return []


def should_patch_kv_inputs(layer_name: str) -> bool:
    if "attn2" in layer_name and (layer_name.endswith(".to_k") or layer_name.endswith(".to_v")):
        return True
    return False
