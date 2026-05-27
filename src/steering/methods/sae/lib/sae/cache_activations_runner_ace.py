import io
import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

from diffusers.utils.import_utils import is_xformers_available
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value, load_dataset, disable_progress_bars, enable_progress_bars
from datasets.fingerprint import generate_fingerprint
from huggingface_hub import HfApi
from tqdm import tqdm
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from src.steering.methods.sae.lib.hooked_model.hooked_model import HookedDiffusionModel
from src.steering.methods.sae.lib.sae.config import CacheActivationsRunnerConfig

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}


class CacheActivationsRunner:
    def __init__(
        self,
        cfg: CacheActivationsRunnerConfig,
        model: HookedDiffusionModel,
        accelerator,
    ):
        self.cfg = cfg
        self.accelerator = accelerator
        self.model = model
        # hacky way to prevent initializing those objects when using only load_and_push_to_hub()
        if self.cfg.hook_names is not None:
            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.model.model.enable_xformers_memory_efficient_attention()
            self.model.model.to(self.accelerator.device)

            self.model.pipeline.music_dcae.to("cpu")
            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}

            self.scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )

            timesteps, _ = retrieve_timesteps(
                self.scheduler,
                num_inference_steps=self.cfg.num_inference_steps,
                device=self.accelerator.device,
                timesteps=None,
            )

            self.scheduler_timesteps = timesteps

            if self.cfg.dataset_type is None or self.cfg.dataset_type == "hf":
                self.dataset = load_dataset(
                    self.cfg.dataset_name,
                    split=self.cfg.split,
                    columns=[self.cfg.column],
                )
                self.dataset = self.dataset.add_column(
                    "prompt_idx", list(range(len(self.dataset)))
                )
            elif self.cfg.dataset_type == "csv":
                self.dataset = Dataset.from_csv(self.cfg.dataset_name).select_columns(
                    [self.cfg.column]
                )
                self.dataset = self.dataset.add_column(
                    "prompt_idx", list(range(len(self.dataset)))
                )
                if self.cfg.dataset_duplicate_rows is not None:
                    indices = [
                        i
                        for i in range(len(self.dataset))
                        for _ in range(self.cfg.dataset_duplicate_rows)
                    ]
                    self.dataset = self.dataset.select(indices)
            self.dataset = self.dataset.shuffle(self.cfg.seed)

            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))

            self.num_examples = len(self.dataset)
            self.dataloader = self.get_batches(self.dataset, self.cfg.batch_size)
            self.n_buffers = len(self.dataloader)

    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,  # buffer shape: "bs num_inference_steps+1 d_sample_size d_in",
        hook_name: str,
        cfg_labels: list,
        prompt_indices: list,
    ) -> Dataset:
        """Create a dataset shard from activations and pre-created labels.

        Args:
            buffer: Activation buffer from gathered GPUs
            hook_name: Name of the hook
            cfg_labels: Pre-created CFG pass labels (already flattened and gathered)
            prompt_indices: Pre-created prompt indices (already flattened and gathered)
        """
        batch_size, n_steps, d_sample_size, d_in = buffer.shape

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)

        # Labels are already created in the correct order, just use them directly
        shard = Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
                "cfg_pass": cfg_labels,
                "prompt_idx": prompt_indices,
            },
            features=self.features_dict[hook_name],
        )
        return shard

    def create_dataset_feature(self, hook_name, d_in, d_out):
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(
                        d_in,
                        d_out,
                    ),
                    dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                ),
                "timestep": Value(dtype="uint16"),
                "cfg_pass": Value(dtype="uint8"),  # 0=cond, 1=uncond, 2=text_only
                "prompt_idx": Value(
                    dtype="uint16"
                ),  # Index of original prompt in batch
            }
        )

    def _compute_num_cfg_passes(self) -> int:
        """Compute number of CFG passes based on guidance settings."""
        do_double_guidance = (
            self.cfg.ace_step_guidance_scale_text is not None
            and self.cfg.ace_step_guidance_scale_text > 1.0
            and self.cfg.ace_step_guidance_scale_lyric is not None
            and self.cfg.ace_step_guidance_scale_lyric > 1.0
        )
        return 3 if do_double_guidance else 2

    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        ### Paths setup
        assert self.cfg.new_cached_activations_path is not None

        final_cached_activation_paths = {
            n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
            for n in self.cfg.hook_names
        }

        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                    )

            tmp_cached_activation_paths = {
                n: path / ".tmp_shards/"
                for n, path in final_cached_activation_paths.items()
            }
            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=False, parents=False)

        self.accelerator.wait_for_everyone()

        # Compute number of CFG passes for labeling
        num_cfg_passes = self._compute_num_cfg_passes()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")
            print(
                f"CFG passes: {num_cfg_passes} ({'double guidance' if num_cfg_passes == 3 else 'standard'})"
            )

        seed = self.cfg.seed

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            seed += 1
            with self.accelerator.split_between_processes(batch) as prompt:
                device_seed = self.accelerator.process_index * 10_000 + seed

                batch_prompt_indices = prompt["prompt_idx"]
                prompt_text = prompt[self.cfg.column]
                local_batch_size = len(prompt_text)
                num_timesteps_to_cache = (
                    self.cfg.num_inference_steps // self.cfg.cache_every_n_timesteps
                )
                local_cfg_labels = []
                local_prompt_indices = []

                for cfg_idx in range(num_cfg_passes):
                    for prompt_idx_in_batch in range(local_batch_size):
                        local_cfg_labels.extend([cfg_idx] * num_timesteps_to_cache)
                        local_prompt_indices.extend(
                            [batch_prompt_indices[prompt_idx_in_batch]]
                            * num_timesteps_to_cache
                        )

                _, acts_cache = self.model.run_with_cache(
                    prompt=prompt_text,
                    return_type="latent",
                    num_inference_steps=self.cfg.num_inference_steps,
                    positions_to_cache=self.cfg.hook_names,
                    guidance_scale=self.cfg.guidance_scale,
                    audio_duration=self.cfg.audio_length_in_s,
                    guidance_scale_text=self.cfg.ace_step_guidance_scale_text,
                    guidance_scale_lyric=self.cfg.ace_step_guidance_scale_lyric,
                    guidance_interval=self.cfg.ace_step_guidance_interval,
                    guidance_interval_decay=self.cfg.ace_step_guidance_interval_decay,
                    manual_seed=device_seed,
                )

            self.accelerator.wait_for_everyone()

            # Gather activations and metadata from all GPUs
            gathered_buffer = {}
            for hook_name in self.cfg.hook_names:
                gathered_buffer[hook_name] = acts_cache["output"][hook_name]
            gathered_buffer = gather_object([gathered_buffer])  # list of dicts

            gathered_cfg_labels = gather_object([local_cfg_labels])
            gathered_prompt_indices = gather_object([local_prompt_indices])

            if self.accelerator.is_main_process:
                # Flatten gathered metadata from all GPUs
                flat_cfg_labels = []
                for gpu_cfg_labels in gathered_cfg_labels:
                    flat_cfg_labels.extend(gpu_cfg_labels)

                flat_prompt_indices = []
                for gpu_indices in gathered_prompt_indices:
                    flat_prompt_indices.extend(gpu_indices)

                for hook_name in self.cfg.hook_names:
                    gathered_buffer_acts = torch.cat(
                        [
                            gathered_buffer[i][hook_name]
                            for i in range(len(gathered_buffer))
                        ],
                        dim=0,
                    )
                    if self.features_dict[hook_name] is None:
                        self.create_dataset_feature(
                            hook_name,
                            gathered_buffer_acts.shape[-2],
                            gathered_buffer_acts.shape[-1],
                        )

                    disable_progress_bars()
                    shard = self._create_shard(
                        gathered_buffer_acts,
                        hook_name,
                        flat_cfg_labels,
                        flat_prompt_indices,
                    )

                    shard.save_to_disk(
                        f"{tmp_cached_activation_paths[hook_name]}/shard_{i:05d}",
                        num_shards=1,
                    )
                    enable_progress_bars()
                    del gathered_buffer_acts, shard
                del gathered_buffer

        ### Concat sharded datasets together, shuffle and push to hub
        datasets = {}

        if self.accelerator.is_main_process:
            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path, final_cached_activation_paths[hook_name], copy_files=False
                )
                print(f"Consolidated the dataset for hook {hook_name}")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                for hook_name, dataset in datasets.items():
                    dataset.push_to_hub(
                        repo_id=f"{self.cfg.hf_repo_id}_{hook_name}",
                        num_shards=self.cfg.hf_num_shards or self.n_buffers,
                        private=self.cfg.hf_is_private_repo,
                        revision=self.cfg.hf_revision,
                    )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )

        return datasets

    def load_and_push_to_hub(self) -> None:
        """Load dataset from disk and push it to the hub."""
        assert self.cfg.new_cached_activations_path is not None
        dataset = Dataset.load_from_disk(self.cfg.new_cached_activations_path)
        if self.accelerator.is_main_process:
            print("Loaded dataset from disk")

            if self.cfg.hf_repo_id:
                print("Pushing to hub...")
                dataset.push_to_hub(
                    repo_id=self.cfg.hf_repo_id,
                    num_shards=self.cfg.hf_num_shards
                    or (len(dataset) // self.cfg.batch_size),
                    private=self.cfg.hf_is_private_repo,
                    revision=self.cfg.hf_revision,
                )

                meta_io = io.BytesIO()
                meta_contents = json.dumps(
                    asdict(self.cfg), indent=2, ensure_ascii=False
                ).encode("utf-8")
                meta_io.write(meta_contents)
                meta_io.seek(0)

                api = HfApi()
                api.upload_file(
                    path_or_fileobj=meta_io,
                    path_in_repo="cache_activations_runner_cfg.json",
                    repo_id=self.cfg.hf_repo_id,
                    repo_type="dataset",
                    commit_message="Add cache_activations_runner metadata",
                )
