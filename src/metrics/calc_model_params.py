import torch

from src.models.audioldm2.patchable_audioldm2 import PatchableAudioLDM2


def get_n_params_from_shape(shape: tuple[int, ...]) -> int:
    n_params = 1
    for dim in shape:
        n_params *= dim
    return n_params


def calc_nparams_and_percent(ca_layers: list[torch.nn.Module], total_params: int) -> dict[str, float | int]:
    n_params = sum(
        [
            get_n_params_from_shape(ca_layer.to_k.weight.shape) + get_n_params_from_shape(ca_layer.to_v.weight.shape)
            for ca_layer in ca_layers
        ]
    )
    percent = n_params / total_params * 100
    return {
        "percent": percent,
        "n_params": n_params,
    }


def main():
    model = PatchableAudioLDM2()

    print(
        f"up1.tf10.attn0: {calc_nparams_and_percent(ca_layers=[model.unet.up_blocks[1].attentions[10].transformer_blocks[0].attn2], total_params=model.unet.num_parameters(only_trainable=True))}"
    )
    print(
        f"up1.tf5.attn0 + up1.tf10.attn0: {calc_nparams_and_percent(ca_layers=[model.unet.up_blocks[1].attentions[5].transformer_blocks[0].attn2, model.unet.up_blocks[1].attentions[10].transformer_blocks[0].attn2], total_params=model.unet.num_parameters(only_trainable=True))}"
    )
    print(
        f"up1.tf5.attn0 + up1.tf5.attn1: {calc_nparams_and_percent(ca_layers=[model.unet.up_blocks[1].attentions[5].transformer_blocks[0].attn2, model.unet.up_blocks[1].attentions[5].transformer_blocks[1].attn2], total_params=model.unet.num_parameters(only_trainable=True))}"
    )
    print(
        f"up1.tf2.attn0 + up1.tf5.attn0 + up1.tf10.attn0: {calc_nparams_and_percent(ca_layers=[model.unet.up_blocks[1].attentions[2].transformer_blocks[0].attn2, model.unet.up_blocks[1].attentions[5].transformer_blocks[0].attn2, model.unet.up_blocks[1].attentions[10].transformer_blocks[1].attn2], total_params=model.unet.num_parameters(only_trainable=True))}"
    )


if __name__ == "__main__":
    main()
