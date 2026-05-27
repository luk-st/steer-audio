import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrograms(
    audio_paths: list[str],
    titles: list[str] | None = None,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> None:
    """
    Plot spectrograms of two audio files stacked vertically with the same value range.

    Args:
        audio_paths: List of paths to audio files
        titles: Optional list of titles for each spectrogram. If None, uses filenames
        sr: Sample rate for loading audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
    """
    # First pass: compute all spectrograms and find global max
    spectrograms = []
    sample_rates = []
    global_max = 0.0

    for audio_path in audio_paths:
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        D_abs = np.abs(D)
        global_max = max(global_max, np.max(D_abs))
        spectrograms.append(D_abs)
        sample_rates.append(sr_loaded)

    # Use filenames as default titles if not provided
    if titles is None:
        titles = [audio_path.split("/")[-1] for audio_path in audio_paths]

    # Second pass: convert to dB with global reference and plot
    fig, axes = plt.subplots(len(audio_paths), 1, figsize=(12, 4 * len(audio_paths)))

    if len(audio_paths) == 1:
        axes = [axes]

    for idx, (title, D_abs, sr_loaded) in enumerate(zip(titles, spectrograms, sample_rates)):
        # Convert to dB using global max as reference
        S_db = librosa.amplitude_to_db(D_abs, ref=global_max)

        # Plot spectrogram with fixed value range
        img = librosa.display.specshow(
            S_db,
            sr=sr_loaded,
            hop_length=hop_length,
            x_axis="time" if idx == len(audio_paths) - 1 else None,
            y_axis="hz",
            ax=axes[idx],
            cmap="viridis",
            vmin=-80,
            vmax=0,
        )

        axes[idx].set_title(title)
        axes[idx].set_ylabel("")

        # Remove x-axis labels for all but the last plot
        if idx < len(audio_paths) - 1:
            axes[idx].set_xlabel("")
            axes[idx].set_xticklabels([])
        else:
            axes[idx].set_xlabel("Time (s)")

    # Add frequency label in the middle of the left side
    fig.text(0.02, 0.5, "Frequency (Hz)", va="center", rotation="vertical", fontsize=12)

    # Add single colorbar on the right side for all subplots
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.92)  # Make room for frequency label and colorbar
    cbar = fig.colorbar(img, ax=axes.tolist() if hasattr(axes, "tolist") else axes, format="%+2.0f dB", pad=0.02)
    cbar.set_label("Power (dB)", rotation=270, labelpad=20)

    plt.show()


def plot_mel_spectrograms(
    audio_paths: list[str],
    titles: list[str] | None = None,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> None:
    """
    Plot mel-spectrograms of two audio files stacked vertically with the same value range.

    Args:
        audio_paths: List of paths to audio files
        titles: Optional list of titles for each spectrogram. If None, uses filenames
        sr: Sample rate for loading audio
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
    """
    # First pass: compute all mel-spectrograms and find global max
    mel_spectrograms = []
    sample_rates = []
    global_max = 0.0

    for audio_path in audio_paths:
        y, sr_loaded = librosa.load(audio_path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        global_max = max(global_max, np.max(S))
        mel_spectrograms.append(S)
        sample_rates.append(sr_loaded)

    # Use filenames as default titles if not provided
    if titles is None:
        titles = [audio_path.split("/")[-1] for audio_path in audio_paths]

    # Second pass: convert to dB with global reference and plot
    fig, axes = plt.subplots(len(audio_paths), 1, figsize=(12, 4 * len(audio_paths)))

    if len(audio_paths) == 1:
        axes = [axes]

    for idx, (title, S, sr_loaded) in enumerate(zip(titles, mel_spectrograms, sample_rates)):
        # Convert to dB using global max as reference
        S_db = librosa.power_to_db(S, ref=global_max)

        # Plot mel-spectrogram with fixed value range
        img = librosa.display.specshow(
            S_db,
            sr=sr_loaded,
            hop_length=hop_length,
            x_axis="time" if idx == len(audio_paths) - 1 else None,
            y_axis="mel",
            ax=axes[idx],
            cmap="viridis",
            vmin=-80,
            vmax=0,
        )

        axes[idx].set_title(title)
        axes[idx].set_ylabel("")

        # Remove x-axis labels for all but the last plot
        if idx < len(audio_paths) - 1:
            axes[idx].set_xlabel("")
            axes[idx].set_xticklabels([])
        else:
            axes[idx].set_xlabel("Time (s)", fontsize=20)

    # Add frequency label in the middle of the left side
    fig.text(0.02, 0.5, "Mel Freq", va="center", rotation="vertical", fontsize=20)

    # Add single colorbar on the right side for all subplots
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.92)  # Make room for frequency label and colorbar
    cbar = fig.colorbar(img, ax=axes.tolist() if hasattr(axes, "tolist") else axes, format="%+2.0f dB", pad=0.02)
    cbar.set_label("Power (dB)", rotation=270, labelpad=20)

    plt.show()


if __name__ == "__main__":
    # Example usage:
    audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    titles = ["Original Audio", "Modified Audio"]

    # With custom titles
    plot_spectrograms(audio_paths, titles=titles)
    plot_mel_spectrograms(audio_paths, titles=titles)

    # Or without titles (will use filenames)
    plot_spectrograms(audio_paths)
    plot_mel_spectrograms(audio_paths)
