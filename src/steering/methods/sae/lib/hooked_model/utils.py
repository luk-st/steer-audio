from typing import List, Optional, Union

import numpy as np
import PIL
import torch
from PIL import Image


def retrieve_sd3(io, unconditional: bool = False):
    if isinstance(io, tuple):
        if len(io) <= 2:
            io = io[0].detach().cpu()
            io_uncond, io_cond = io.chunk(2)
            if unconditional:
                return io_uncond
            return io_cond
        else:
            raise ValueError("A tuple should have length of 1 or 2")
    elif isinstance(io, torch.Tensor):
        io = io.detach().cpu()
        io_uncond, io_cond = io.chunk(2)
        if unconditional:
            return io_uncond
        return io_cond
    else:
        raise ValueError("Input/Output must be a tensor, or 1/2-element tuple")


def retrieve(io, unconditional: bool = False):
    if isinstance(io, tuple):
        if len(io) == 1:
            io = io[0].detach().cpu()
            io_uncond, io_cond = io.chunk(2)
            if unconditional:
                return io_uncond
            return io_cond
        else:
            raise ValueError("A tuple should have length of 1")
    elif isinstance(io, torch.Tensor):
        io = io.detach().cpu()
        io_uncond, io_cond = io.chunk(2)
        if unconditional:
            return io_uncond
        return io_cond
    else:
        raise ValueError("Input/Output must be a tensor, or 1-element tuple")


def locate_block(position: str, model: torch.nn.Module):
    """
    Locate a specific block in the model given its position string.
    Handles both attribute access and indexed access.
    """
    block = model
    for step in position.split("."):
        if step.isdigit():
            step = int(step)
            block = block[step]
        else:
            block = getattr(block, step)
    return block


def get_timesteps(
    num_inference_steps: int,
    num_train_timesteps: int,
    timestep_spacing: str = "leading",
    steps_offset: int = 0,
    device: Union[str, torch.device] = None,
) -> torch.Tensor:
    """
    Gets the discrete timesteps used for the diffusion chain.

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
        num_train_timesteps (`int`):
            The number of timesteps used to train the model.
        timestep_spacing (`str`, optional):
            How to space the timesteps. One of "linspace", "leading", or "trailing". Defaults to "leading".
        steps_offset (`int`, optional):
            Offset added to timesteps for "leading" spacing. Defaults to 0.
        device (`Union[str, torch.device]`, optional):
            The device to put the timesteps on.

    Returns:
        torch.Tensor: The timesteps tensor
    """

    if num_inference_steps > num_train_timesteps:
        raise ValueError(
            f"`num_inference_steps`: {num_inference_steps} cannot be larger than `num_train_timesteps`:"
            f" {num_train_timesteps} as the unet model trained with this scheduler can only handle"
            f" maximal {num_train_timesteps} timesteps."
        )

    # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
    if timestep_spacing == "linspace":
        timesteps = (
            np.linspace(0, num_train_timesteps - 1, num_inference_steps)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
    elif timestep_spacing == "leading":
        step_ratio = num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps += steps_offset
    elif timestep_spacing == "trailing":
        step_ratio = num_train_timesteps / num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = np.round(np.arange(num_train_timesteps, 0, -step_ratio)).astype(
            np.int64
        )
        timesteps -= 1
    else:
        raise ValueError(
            f"{timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
        )

    return torch.from_numpy(timesteps).to(device)


def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    r"""
    Convert a PyTorch tensor to a NumPy image.

    Args:
        images (`torch.Tensor`):
            The PyTorch tensor to convert to NumPy format.

    Returns:
        `np.ndarray`:
            A NumPy array representation of the images.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images


def numpy_to_pil(images: np.ndarray) -> List[PIL.Image.Image]:
    r"""
    Convert a numpy image or a batch of images to a PIL image.

    Args:
        images (`np.ndarray`):
            The image array to convert to PIL format.

    Returns:
        `List[PIL.Image.Image]`:
            A list of PIL images.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def denormalize(
    images: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    r"""
    Denormalize an image array to [0,1].

    Args:
        images (`np.ndarray` or `torch.Tensor`):
            The image array to denormalize.

    Returns:
        `np.ndarray` or `torch.Tensor`:
            The denormalized image array.
    """
    return (images * 0.5 + 0.5).clamp(0, 1)


def denormalize_conditionally(
    images: torch.Tensor, do_denormalize: Optional[List[bool]] = None
) -> torch.Tensor:
    r"""
    Denormalize a batch of images based on a condition list.

    Args:
        images (`torch.Tensor`):
            The input image tensor.
        do_denormalize (`Optional[List[bool]`, *optional*, defaults to `None`):
            A list of booleans indicating whether to denormalize each image in the batch. If `None`, will use the
            value of `do_normalize` in the `VaeImageProcessor` config.
    """
    if do_denormalize is None:
        return images

    return torch.stack(
        [
            denormalize(images[i]) if do_denormalize[i] else images[i]
            for i in range(images.shape[0])
        ]
    )


def postprocess_image(
    image: torch.Tensor,
    output_type: str = "pil",
    do_denormalize: Optional[List[bool]] = None,
) -> Union[PIL.Image.Image, np.ndarray, torch.Tensor]:
    """
    Postprocess the image output from tensor to `output_type`.

    Args:
        image (`torch.Tensor`):
            The image input, should be a pytorch tensor with shape `B x C x H x W`.
        output_type (`str`, *optional*, defaults to `pil`):
            The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.

    Returns:
        `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
            The postprocessed image.
    """
    if output_type == "latent":
        return image

    image = denormalize_conditionally(image, do_denormalize)

    image = pt_to_numpy(image)

    if output_type == "np":
        return image

    if output_type == "pil":
        return numpy_to_pil(image)
