import shutil
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import time

SAMPLE_RATE = 44100


def load_audios_save(path: Path | str, max_workers: int = 1, sample_rate: int = SAMPLE_RATE) -> Path:
    """Loads audios from a numpy file and saves them as wav files."""

    def save_audio(idx_audio_pair):
        idx, audio = idx_audio_pair
        torchaudio.save((dir_path / f"a_id{idx}.wav").resolve(), audio, sample_rate=sample_rate)

    if isinstance(path, str):
        path = Path(path)
    assert path.suffix == ".npy"
    dir_path = (path.parent / "audios").resolve()
    dir_path.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        # find all .npy files that are named xxx_{int}.npy
        npy_files = list(path.parent.glob(path.stem + "_*.npy"))
        npy_files = sorted(npy_files, key=lambda x: int(x.stem.split("_")[-1]))
    else:
        npy_files = [path]

    idx = 0
    all_audio_tasks = []

    for npy_path in npy_files:
        audios = np.load(npy_path)
        audios_tensor = torch.from_numpy(audios)
        if len(audios_tensor.shape) == 2:
            audios_tensor = audios_tensor.unsqueeze(1)

        for audio in audios_tensor:
            all_audio_tasks.append((idx, audio))
            idx += 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(save_audio, all_audio_tasks),
                total=len(all_audio_tasks),
                desc=f"Saving audios: {dir_path}",
            )
        )

    return dir_path


def del_audios_dir(path: Path) -> None:
    """Deletes the audios directory."""
    print(f"Deleting audios directory: {path}")
    shutil.rmtree(path)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--sample_rate", type=int, default=44100)
    args = parser.parse_args()

    load_audios_save(args.path, args.max_workers, args.sample_rate)