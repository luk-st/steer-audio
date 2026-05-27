from datetime import datetime
from pathlib import Path

import fire
from datasets import load_dataset
from stable_audio_run import run_stable_audio_edit
from tqdm import tqdm

from utils import set_reproducability

PATHS_MUSICCAPS = "/data/lstaniszewski/code/audio-interv/data/music_caps_shorted"
EDITING_TYPE_ID_DEFAULT = "0"  # '5'
PATH_DIR_OUTPUT = Path("/data/lstaniszewski/code/audio-interv/editing/outputs").resolve()

dt_str = datetime.now().strftime("%Y%m%d%H%M%S")


def prepare_dataset(editing_type_id: str, path_musiccaps: Path):
    if isinstance(editing_type_id, int):
        editing_type_id = str(editing_type_id)
    ds = load_dataset("liuhuadai/ZoME-Bench")
    df_train = ds["train"].to_pandas()
    df_instruments = df_train[df_train["editing_type_id"] == editing_type_id]
    path_musiccaps = Path(path_musiccaps)
    filenames_paths = [p for p in path_musiccaps.glob("*.wav")]
    shorted_filenames = [p.name.split("]")[0][1:] for p in filenames_paths]
    name_to_path = {sfn: fnp for sfn, fnp in zip(shorted_filenames, filenames_paths)}
    df_instruments = df_instruments[df_instruments["ytid"].isin(shorted_filenames)]
    df_instruments["path_yt"] = df_instruments.apply(lambda x: name_to_path.get(x["ytid"], None), axis=1)
    return df_instruments


def main(
    editing_type_id: str = EDITING_TYPE_ID_DEFAULT,
    dataset_name: str = "zome",
    mode: str = "ddpm",
    num_diffusion_steps: int = 100,
    cfg_src: float = 1.0,
    cfg_tar: float = 3.5,
    tstart: int = 50,
    target_neg_prompt: str = "",
    seed: int = 42,
    run_name: str | None = None,
    layers_to_hook: str | list[str] | None = None,
):
    """
    Main function to edit audio files using Stable Audio methods.

    Args:
        editing_type_id: Type of editing to perform (default: "0")
        dataset_name: Name of the dataset (default: "zome")
        mode: Editing mode - 'ddpm', 'ddim', or 'sdedit' (default: "ddpm")
        num_diffusion_steps: Number of diffusion steps (default: 100)
        cfg_src: Classifier-free guidance strength for forward process (default: [1.0])
        cfg_tar: Classifier-free guidance strength for reverse process (default: [3.5])
        tstart: Diffusion timestep to start the reverse process from (default: 50)
        target_neg_prompt: Negative prompt for target generation (default: "")
        cutoff_points: Cutoff points for DDPM mode (default: None)
        fix_alpha: Alpha parameter for DDPM mode (default: 0.1)
        eta: Eta parameter for DDPM mode (default: 1.0)
        numerical_fix: Whether to apply numerical fix (default: True)
        seed: Random seed (default: 42)
        run_name: Custom run name (default: None - auto-generated)
        layers_to_hook: Layer(s) to hook for cross-attention replacement - can be a string for single layer or list for multiple layers (default: None)

    Returns:
        None
    """
    # Convert layers_to_hook to proper format
    if isinstance(layers_to_hook, str):
        if "," in layers_to_hook:
            layers_to_hook = layers_to_hook.split(",")
        else:
            layers_to_hook = [layers_to_hook]
    elif layers_to_hook is None:
        # If None, use empty list (no hooking)
        layers_to_hook = []
    # Validate mode
    assert mode in ["ddpm", "ddim", "sdedit"], "Invalid mode, must be one of ['ddpm', 'ddim', 'sdedit']"

    # Prepare dataset
    df_instruments = prepare_dataset(editing_type_id=editing_type_id, path_musiccaps=Path(PATHS_MUSICCAPS))

    # Generate run name if not provided
    if run_name is None:
        dt_str = datetime.now().strftime("%Y%m%d%H%M%S")
        cfg_src_str = "-".join([str(x) for x in cfg_src])
        cfg_tar_str = "-".join([str(x) for x in cfg_tar])
        run_name = f"stable_audio_{mode}_{dt_str}_cfg_src{cfg_src_str}_cfg_tar{cfg_tar_str}_tstart{tstart}_steps{num_diffusion_steps}"

    # Create output directory
    path_dir_outs = (PATH_DIR_OUTPUT / f"{dataset_name}_{editing_type_id}" / run_name).resolve()
    path_dir_outs.mkdir(parents=True, exist_ok=True)

    set_reproducability(seed, extreme=False)

    print(f"Processing {len(df_instruments)} audio files...")
    print(f"Mode: {mode}")
    print(f"Output directory: {path_dir_outs}")

    # Process each audio file in the dataset
    with tqdm(total=len(df_instruments), desc="Processing files") as pbar:
        for idx, row in df_instruments.iterrows():
            path_to_audio = str(row["path_yt"])
            source_prompt = row["original_prompt"]
            target_prompt = row["editing_prompt"]

            # Create output path for this specific file
            output_wav_path = str(path_dir_outs / f"a{idx}.wav")

            # Call the original run_stable_audio_edit function with save_edit_wav_path
            run_stable_audio_edit(
                init_aud=path_to_audio,
                cfg_src=[cfg_src],
                cfg_tar=[cfg_tar],
                num_diffusion_steps=num_diffusion_steps,
                target_prompt=[target_prompt],
                source_prompt=[source_prompt],
                target_neg_prompt=[target_neg_prompt],
                tstart=[tstart],
                results_path="",  # Not used when save_edit_wav_path is provided
                mode=mode,
                verbose=False,  # Disable individual verbose to avoid spam
                save_edit_wav_path=output_wav_path,
                layers_to_hook=layers_to_hook,
            )
            pbar.update(1)

    print(f"\nCompleted processing {len(df_instruments)} files.")
    print(f"Results saved to: {path_dir_outs}")


if __name__ == "__main__":
    fire.Fire(main)
