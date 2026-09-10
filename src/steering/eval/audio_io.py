
from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf  # libsndfile: reads/writes PCM wavs without FFmpeg/torchcodec
import torch

AUDIOS_NPZ = "audios.npz"  # one file per alpha dir, replaces p0.wav..pN.wav


def _to_int16(audio: torch.Tensor) -> np.ndarray:
    """[C,T] (or [T]) float in [-1,1] -> int16 PCM, matching torchaudio.save(PCM_16)."""
    a = audio if torch.is_tensor(audio) else torch.as_tensor(audio)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    a = (a.detach().cpu().float() * 32768.0).round().clamp(-32768.0, 32767.0)
    return a.to(torch.int16).numpy()


def save_alpha_audios(
    alpha_dir: str | Path,
    audios: List[torch.Tensor],
    sr: int,
    names: Optional[List[str]] = None,
    compress: bool = False,
) -> Path:
    """Pack a list of [C,T] float audios into ``alpha_dir/audios.npz`` (int16 PCM).

    ``names`` preserves per-clip identity/order (defaults p0..pN); the loader returns
    them sorted, so index-based pairing (e.g. LPAPS steered[i] vs baseline[i]) is stable.
    ``compress=False`` (default) is much faster to write/read; int16 audio only shrinks
    ~10% under deflate, and the goal here is fewer files, not smaller ones.
    """
    alpha_dir = Path(alpha_dir)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    arrs = [_to_int16(a) for a in audios]
    assert arrs, f"no audios to save for {alpha_dir}"
    shapes = {a.shape for a in arrs}
    assert len(shapes) == 1, f"ragged audio shapes in {alpha_dir}: {shapes}"
    audio = np.stack(arrs, axis=0)  # [N, C, T]
    if names is None:
        names = [f"p{i}.wav" for i in range(len(arrs))]
    saver = np.savez_compressed if compress else np.savez
    saver(alpha_dir / AUDIOS_NPZ, audio=audio, sr=np.int64(sr), names=np.array(names))
    return alpha_dir / AUDIOS_NPZ


def save_alpha_int16(
    alpha_dir: str | Path,
    audio_int16: np.ndarray,
    sr: int,
    names: List[str],
    compress: bool = False,
) -> Path:
    """Write an already-int16 ``[N, C, T]`` stack straight to ``audios.npz`` (low memory).

    Used by migration: reading wavs as int16 and storing them verbatim avoids the float32
    round-trip (which is 2x the RAM) and is bit-exact with the source PCM.
    """
    assert audio_int16.dtype == np.int16, audio_int16.dtype
    alpha_dir = Path(alpha_dir)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compress else np.savez
    saver(alpha_dir / AUDIOS_NPZ, audio=audio_int16, sr=np.int64(sr), names=np.array(names))
    return alpha_dir / AUDIOS_NPZ


def has_packed(alpha_dir: str | Path) -> bool:
    return (Path(alpha_dir) / AUDIOS_NPZ).exists()


def load_alpha_audios(alpha_dir: str | Path) -> Tuple[List[torch.Tensor], Optional[int], List[str]]:
    """Return ([C,T] float32 tensors in [-1,1], sr, names), sorted by name.

    Reads ``audios.npz`` if present (int16/32768 == torchaudio.load), else falls back to
    the legacy ``*.wav`` files, so packed and unpacked data both work.
    """
    alpha_dir = Path(alpha_dir)
    npz = alpha_dir / AUDIOS_NPZ
    if npz.exists():
        d = np.load(npz, allow_pickle=False)
        arr, sr = d["audio"], int(d["sr"])
        names = [str(n) for n in d["names"]]
        order = np.argsort(names)
        audios = [torch.from_numpy(arr[i].astype(np.float32) / 32768.0) for i in order]
        return audios, sr, [names[i] for i in order]
    wavs = sorted(alpha_dir.glob("*.wav"))
    audios, sr = [], None
    for w in wavs:
        data, s = sf.read(str(w), dtype="float32", always_2d=True)  # [T, C], int16/32768
        audios.append(torch.from_numpy(np.ascontiguousarray(data.T)))  # -> [C, T]
        sr = sr or s
    return audios, sr, [w.name for w in wavs]


@contextlib.contextmanager
def as_wav_dir(alpha_dir: str | Path):
    """Yield a directory of ``p{i}.wav`` for path-based tools (MuQ/CLAP).

    If the dir already has wavs, yields it unchanged (zero cost). If only ``audios.npz``
    exists, extracts to a temp dir (bit-exact int16) and cleans up afterwards.
    """
    alpha_dir = Path(alpha_dir)
    if any(alpha_dir.glob("*.wav")):
        yield alpha_dir
        return
    audios, sr, names = load_alpha_audios(alpha_dir)
    tmp = Path(tempfile.mkdtemp(prefix="wavdir_"))
    try:
        for a, n in zip(audios, names):
            n = n if n.endswith(".wav") else f"{n}.wav"
            # a is float [-1,1] == int16/32768; write PCM_16 (round(a*32768) recovers
            # the original int16 samples bit-exactly). soundfile wants [T] or [T, C].
            arr = a.detach().cpu().numpy().T  # [T, C]
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            sf.write(str(tmp / n), arr, sr, subtype="PCM_16")
        yield tmp
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
