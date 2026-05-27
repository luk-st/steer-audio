"""
Train sparse autoencoders on activations from a diffusion model.
"""

import os
import sys
from contextlib import nullcontext, redirect_stdout
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets
from simple_parsing import parse

from src.steering.methods.sae.lib.sae.config import SaeConfig, TrainConfig
from src.steering.methods.sae.lib.sae.trainer import SaeTrainer


@dataclass
class ALDMSaeConfig(SaeConfig):
    expansion_factor: int = 4
    batch_topk: bool = True
    k: int = 24

@dataclass
class ALDMTrainConfig(TrainConfig):
    sae: ALDMSaeConfig
    num_workers: int = 48
    lr: float = 3e-5
    lr_scheduler: str = "linear"
    auxk_alpha: float = 0.03125
    wandb_log_frequency: int = 1000
    wandb_project: str = "music_sae"
    lr_warmup_steps: int = 1000

@dataclass
class ALDMRunConfig(ALDMTrainConfig):
    mixed_precision: str = "no"
    max_examples: int | None = None
    seed: int = 42
    device: str = "cuda"
    num_epochs: int = 5


def load_datasets_from_dirs(base_dirs, hookpoint, dtype=torch.float32):
    """
    Load and concatenate datasets from multiple directories.

    Args:
        base_dirs (list[str]): List of base directory paths containing the datasets
        hookpoint (str): Name of the hookpoint directory
        dtype: Data type for the tensors (default: torch.float32)

    Returns:
        Dataset: Concatenated dataset
    """
    datasets = []
    print(f"Concatenating datasets from {base_dirs}")

    for base_dir in base_dirs:
        dataset = Dataset.load_from_disk(os.path.join(base_dir, hookpoint), keep_in_memory=False)

        # Set format for each dataset
        dataset.set_format(
            type="torch",
            columns=["activations", "timestep"],
            dtype=dtype,
        )

        datasets.append(dataset)

    # Concatenate all datasets
    return concatenate_datasets(datasets)


def run():
    local_rank = os.environ.get("LOCAL_RANK")
    ddp = local_rank is not None
    rank = int(local_rank) if ddp else 0

    if ddp:
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group("nccl")

        if rank == 0:
            print(f"Using DDP across {dist.get_world_size()} GPUs.")

    args = parse(ALDMRunConfig)
    # add output_or_diff to the run name
    args.run_name = args.run_name + f"_{args.dataset_path[0].split('/')[-2]}"

    dtype = torch.float32
    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    args.dtype = dtype
    if not ddp or rank == 0:
        print(f"Training in {dtype=}")
        print(args)
    # Awkward hack to prevent other ranks from duplicating data preprocessing
    dataset_dict = {}
    if not ddp or rank == 0:
        for hookpoint in args.hookpoints:
            if len(args.dataset_path) > 1:
                dataset = load_datasets_from_dirs(args.dataset_path, hookpoint, dtype)
            else:
                dataset = Dataset.load_from_disk(os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False)
            dataset.set_format(
                type="torch",
                columns=["activations", "timestep"],
                dtype=dtype,
            )
            dataset = dataset.shuffle(args.seed)
            if limit := args.max_examples:
                dataset = dataset.select(range(limit))
            dataset_dict[hookpoint] = dataset
            print(f"Loaded dataset for {hookpoint}")
    # NOTE: DDP not tested so far
    if ddp:
        dist.barrier()
        if rank != 0:
            for hookpoint in args.hookpoints:
                dataset = Dataset.load_from_disk(os.path.join(args.dataset_path[0], hookpoint), keep_in_memory=False)
                dataset.set_format(
                    type="torch",
                    columns=["activations", "timestep"],
                    dtype=dtype,
                )
                dataset = dataset.shuffle(args.seed)
                dataset = dataset.shard(dist.get_world_size(), rank)
                dataset_dict[hookpoint] = dataset
                print(f"Loaded dataset for {hookpoint}")

    # Prevent ranks other than 0 from printing
    with nullcontext() if rank == 0 else redirect_stdout(None):
        trainer = SaeTrainer(args, dataset_dict)

        trainer.fit()


if __name__ == "__main__":
    run()
