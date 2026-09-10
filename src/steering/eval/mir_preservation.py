# ABOUTME: Decomposed MIR preservation distances (harmony/rhythm/melody/structure) between
# ABOUTME: steered(alpha) and baseline(alpha_0) audio, written as lpaps.csv-shaped per-axis CSVs.

import faulthandler
import warnings
from pathlib import Path

import fire
import librosa
import numpy as np
import pandas as pd

faulthandler.enable()

AXES = ("harmony", "rhythm", "melody", "ssm")


def _alpha_dirs(cell: Path) -> dict[float, Path]:
    """Map alpha value -> alpha_* dir for one eval cell."""
    out = {}
    for d in cell.glob("alpha_*"):
        if d.is_dir():
            out[round(float(d.name[len("alpha_") :]), 4)] = d
    return out


def _load_alpha_int16(d: Path) -> tuple[dict[str, np.ndarray], int]:
    """Load one alpha dir as {name: [C,T] int16}, sr. Reads packed audios.npz
    (the benchmark format) or legacy per-prompt ``p*.wav`` files."""
    npz = d / "audios.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=False)
        arr, sr = z["audio"], int(z["sr"])
        assert arr.ndim == 3 and arr.dtype == np.int16, (arr.shape, arr.dtype)
        return {str(n): arr[i] for i, n in enumerate(z["names"])}, sr
    import soundfile as sf

    clips, sr = {}, None
    for w in sorted(d.glob("*.wav")):
        data, s = sf.read(str(w), dtype="int16", always_2d=True)  # [T, C]
        clips[w.name] = np.ascontiguousarray(data.T)
        sr = sr or s
    assert clips, f"no audios.npz or *.wav in {d}"
    return clips, sr


def _prep(clip: np.ndarray, sr_in: int, sr: int) -> np.ndarray:
    """[C,T] int16 -> mono float32 in [-1,1] resampled to `sr` (channel mean, soxr_hq)."""
    assert clip.ndim == 2, clip.shape
    y = clip.astype(np.float32).mean(axis=0) / 32768.0
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr) if sr_in != sr else y


def _chroma(y: np.ndarray, sr: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.feature.chroma_cqt(y=y, sr=sr)  # (12, T)


def _clip_features(clip: np.ndarray, sr_in: int, sr: int, axes: list) -> dict:
    """Per-clip features for the requested axes (the expensive, parallelized stage);
    each pairwise distance below is computed from these alone, so every clip —
    baseline included — is analyzed exactly once."""
    y = _prep(clip, sr_in, sr)
    out = {}
    if "harmony" in axes or "ssm" in axes:
        out["chroma"] = _chroma(y, sr)
    if "rhythm" in axes:
        _, frames = librosa.beat.beat_track(y=y, sr=sr)
        out["beats"] = librosa.frames_to_time(frames, sr=sr)
    if "melody" in axes:
        fmin, fmax = librosa.note_to_hz("C2"), librosa.note_to_hz("C7")
        out["f0"] = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)[0]
    return out


def _harmony(cb: np.ndarray, cs: np.ndarray) -> float:
    """1 - mean per-frame cosine similarity between two chromagrams."""
    T = min(cb.shape[1], cs.shape[1])
    b, s = cb[:, :T], cs[:, :T]
    num = (b * s).sum(0)
    den = np.linalg.norm(b, axis=0) * np.linalg.norm(s, axis=0) + 1e-9
    return 1.0 - float((num / den).mean())


def _ssm(cb: np.ndarray, cs: np.ndarray) -> float:
    """1 - max(Pearson corr, 0) of the two chroma SSMs (upper triangle), so the axis is
    bounded in [0, 1]. Structurally unrelated clips already sit at r ~ 0, i.e. distance 1;
    r < 0 does not mean "worse than unrelated" but a degenerate, near-constant SSM whose
    correlation is computed over numerical noise, so it is clamped to the same 1."""
    T = min(cb.shape[1], cs.shape[1])
    b = cb[:, :T] / (np.linalg.norm(cb[:, :T], axis=0) + 1e-9)
    s = cs[:, :T] / (np.linalg.norm(cs[:, :T], axis=0) + 1e-9)
    iu = np.triu_indices(T, k=1)
    sb, ss = (b.T @ b)[iu], (s.T @ s)[iu]
    r = np.corrcoef(sb, ss)[0, 1]
    return 1.0 - max(float(r) if np.isfinite(r) else 0.0, 0.0)


def _rhythm(bt: np.ndarray, st: np.ndarray) -> float:
    """1 - mir_eval beat F-measure between baseline and steered beat times."""
    import mir_eval

    if len(bt) == 0 or len(st) == 0:
        return float("nan")
    return 1.0 - float(mir_eval.beat.f_measure(bt, st))


def _melody(fb: np.ndarray, fs: np.ndarray) -> float:
    """1 - raw pitch accuracy of the steered f0 contour vs baseline (pyin, monophonic proxy)."""
    import mir_eval

    T = min(len(fb), len(fs))
    fb, fs = fb[:T], fs[:T]
    vb, vs = ~np.isnan(fb), ~np.isnan(fs)
    cb = mir_eval.melody.hz2cents(np.nan_to_num(fb))
    cs = mir_eval.melody.hz2cents(np.nan_to_num(fs))
    return 1.0 - float(mir_eval.melody.raw_pitch_accuracy(vb, cb, vs, cs))


def _pair_metrics(fb: dict, fs: dict, axes: list) -> dict:
    """All requested axis distances for one (baseline, steered) feature pair (cheap stage)."""
    out = {}
    if "harmony" in axes:
        out["harmony"] = _harmony(fb["chroma"], fs["chroma"])
    if "ssm" in axes:
        out["ssm"] = _ssm(fb["chroma"], fs["chroma"])
    if "rhythm" in axes:
        out["rhythm"] = _rhythm(fb["beats"], fs["beats"])
    if "melody" in axes:
        out["melody"] = _melody(fb["f0"], fs["f0"])
    return out


def main(
    cell_dir: str,
    out_root: str = "output/mir_preservation",
    axes: str = "harmony,rhythm,melody,ssm",
    n_prompts: int | None = None,
    sr: int = 22050,
    n_jobs: int = -1,
    align_subdir: str = "protocol_results",
    smoke: bool = False,
):
    """Write per-axis lpaps.csv-shaped preservation distances for one eval cell.

    For each alpha, each requested MIR axis distance is computed per prompt against
    the alpha_0 baseline audio (paired by prompt name), then meaned into
    `<out_root>/<cell>/protocol_results_<axis>/lpaps.csv` (columns alpha,mean,std) so
    the existing auc.py can integrate it unchanged. The muqt/clap alignment CSVs are
    copied in so the table scripts find them. Writes only under out_root (never into
    the read-only cell dir).

    Args:
        cell_dir: eval cell dir, e.g. pwr-mount/outputs/eval/caa_loc_piano.
        out_root: local output root; results go under <out_root>/<cell_name>/.
        axes: comma-separated subset of harmony,rhythm,melody,ssm.
        n_prompts: use only the first N prompts (default: all).
        sr: MIR analysis sample rate (mono).
        n_jobs: joblib workers over clips (-1 = all cores).
        align_subdir: results subdir to copy muqt/clap alignment CSVs from
            (default "protocol_results", the paper pass for every concept).
        smoke: 2 prompts, 2 non-zero alphas, print first distances loudly.
    """
    if isinstance(axes, (list, tuple)):
        axes = [str(a) for a in axes]
    else:
        axes = [a for a in str(axes).split(",") if a]
    assert set(axes) <= set(AXES), f"unknown axes: {set(axes) - set(AXES)}"

    cell = Path(cell_dir)
    adirs = _alpha_dirs(cell)
    assert 0.0 in adirs, f"no alpha_0 baseline in {cell}"
    base_clips, base_sr = _load_alpha_int16(adirs[0.0])
    np_ = 2 if smoke else n_prompts
    names = sorted(base_clips)[:np_] if np_ else sorted(base_clips)

    alphas = sorted(a for a in adirs if a != 0.0)
    if smoke:
        alphas = alphas[:2]
    c0 = base_clips[names[0]]
    print(
        f"[{cell.name}] {len(alphas)} alphas x {len(names)} prompts, axes={axes}, "
        f"clip {c0.shape} {c0.dtype} @ {base_sr}Hz "
        f"({c0.shape[-1] / base_sr:.1f}s) -> mono {sr}Hz, n_jobs={1 if smoke else n_jobs}",
        flush=True,
    )

    from joblib import Parallel, delayed

    # librosa's numba gufuncs are cache=True; concurrent first-use compilation by the
    # workers races on the shared on-disk cache (corrupt entries SIGSEGV at dispatch).
    # Compile everything once here so workers only ever read a complete cache.
    _clip_features(c0[:, : 2 * base_sr], base_sr, sr, axes)
    print("numba cache warmed in driver", flush=True)

    rows = {ax: [(0.0, 0.0, 0.0)] for ax in axes}  # alpha_0 self-distance is exactly 0
    with Parallel(n_jobs=1 if smoke else n_jobs, verbose=10) as par:
        base_feats = dict(
            zip(
                names,
                par(
                    delayed(_clip_features)(base_clips[n], base_sr, sr, axes)
                    for n in names
                ),
            )
        )
        del base_clips
        for a in alphas:
            clips, csr = _load_alpha_int16(adirs[a])
            missing = set(names) - set(clips)
            assert not missing, f"alpha_{a}: missing prompts {sorted(missing)[:5]}"
            feats = par(delayed(_clip_features)(clips[n], csr, sr, axes) for n in names)
            dists = [
                _pair_metrics(base_feats[n], f, axes) for n, f in zip(names, feats)
            ]
            means = {}
            for ax in axes:
                v = np.array([d[ax] for d in dists], float)
                rows[ax].append((a, np.nanmean(v), np.nanstd(v)))
                means[ax] = np.nanmean(v)
            print(
                f"  alpha={a:+.2f}  "
                + "  ".join(f"{ax}={means[ax]:.4f}" for ax in axes),
                flush=True,
            )

    import shutil

    for ax in axes:
        rdir = Path(out_root) / cell.name / f"protocol_results_{ax}"
        rdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows[ax], columns=["alpha", "mean", "std"]).sort_values(
            "alpha"
        ).to_csv(rdir / "lpaps.csv", index=False)
        for f in ("muqt.csv", "clap.csv"):
            src = cell / align_subdir / f
            assert src.exists(), f"missing alignment CSV {src}"
            shutil.copy(src, rdir / f)
    print(f"wrote {axes} for {cell.name} -> {Path(out_root) / cell.name}")


if __name__ == "__main__":
    fire.Fire(main)
