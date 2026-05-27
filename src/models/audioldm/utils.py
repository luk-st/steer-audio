import re


def get_cross_attention_inputs_keys(layer_name: str) -> list[str]:
    pattern_down_blocks = re.compile(r"\.unet\.down_blocks\.\d+$")
    pattern_mid_blocks = re.compile(r"\.unet\.mid_block$")
    pattern_up_blocks = re.compile(r"\.unet\.up_blocks\.\d+$")
    pattern_down_blocks_tfs = re.compile(r"\.unet\.down_blocks\.\d+\.attentions\.\d+$")
    pattern_mid_blocks_tfs = re.compile(r"\.unet\.mid_block\.attentions\.\d+$")
    pattern_up_blocks_tfs = re.compile(r"\.unet\.up_blocks\.\d+\.attentions\.\d+$")
    pattern_down_blocks_tfs_attns = re.compile(r"\.unet\.down_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+$")
    pattern_mid_blocks_tfs_attns = re.compile(r"\.unet\.mid_block\.attentions\.\d+\.transformer_blocks\.\d+$")
    pattern_up_blocks_tfs_attns = re.compile(r"\.unet\.up_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+$")
    pattern_down_blocks_tfs_attns2 = re.compile(
        r"\.unet\.down_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn2$"
    )
    pattern_mid_blocks_tfs_attns2 = re.compile(r"\.unet\.mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn2$")
    pattern_up_blocks_tfs_attns2 = re.compile(
        r"\.unet\.up_blocks\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn2$"
    )

    result = ["encoder_hidden_states"]
    if layer_name == ".unet" or (
        pattern_down_blocks.match(layer_name)
        or pattern_mid_blocks.match(layer_name)
        or pattern_up_blocks.match(layer_name)
    ):
        result.append("encoder_hidden_states_1")
        result.append("encoder_attention_mask_1")
    elif (
        pattern_down_blocks_tfs.match(layer_name)
        or pattern_mid_blocks_tfs.match(layer_name)
        or pattern_up_blocks_tfs.match(layer_name)
        or pattern_down_blocks_tfs_attns.match(layer_name)
        or pattern_mid_blocks_tfs_attns.match(layer_name)
        or pattern_up_blocks_tfs_attns.match(layer_name)
    ):
        result.append("encoder_attention_mask")
    elif (
        pattern_down_blocks_tfs_attns2.match(layer_name)
        or pattern_mid_blocks_tfs_attns2.match(layer_name)
        or pattern_up_blocks_tfs_attns2.match(layer_name)
    ):
        result.append("attention_mask")
    return result
