from __future__ import annotations

from pathlib import Path
from typing import Tuple
import tempfile

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Try to import pydub, but handle Python 3.13 compatibility issues
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _validate_audio_input(audio: np.ndarray, sample_rate: int) -> None:
    """Validate audio input parameters with clear error messages."""
    if audio is None or not isinstance(audio, np.ndarray):
        raise ValueError("'audio' must be a numpy array")
    
    if audio.dtype != np.float32:
        raise ValueError(f"'audio' must be float32, got {audio.dtype}")
    
    if audio.ndim != 1:
        raise ValueError(f"'audio' must be 1-dimensional, got {audio.ndim}D")
    
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"'sample_rate' must be a positive integer, got {sample_rate}")


def save_wav(audio: np.ndarray, sample_rate: int, out_path: Path) -> Path:
    """Save mono float32 audio to a WAV file.

    Parameters
    ----------
    audio: np.ndarray
        1-D float32 numpy array in [-1, 1].
    sample_rate: int
        Sampling rate in Hz.
    out_path: Path
        Destination path for the WAV file.
    """
    _validate_audio_input(audio, sample_rate)

    _ensure_parent_dir(out_path)
    sf.write(str(out_path), audio.astype(np.float32), samplerate=sample_rate, subtype="PCM_16")
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise IOError(f"Failed to write WAV file at {out_path}")
    return out_path


def save_mp3(audio: np.ndarray, sample_rate: int, out_path: Path) -> Path:
    """Save audio to MP3 via pydub/ffmpeg.

    This writes a temporary WAV first and converts to MP3 using pydub.
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError(
            "MP3 export not available. pydub is not compatible with Python 3.13. "
            "Use WAV format instead or upgrade to a compatible Python version."
        )
    
    _validate_audio_input(audio, sample_rate)

    _ensure_parent_dir(out_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        save_wav(audio, sample_rate, tmp_wav)
        seg = AudioSegment.from_wav(str(tmp_wav))
        seg.export(str(out_path), format="mp3")
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise IOError(f"Failed to write MP3 file at {out_path}")
    return out_path


def plot_waveform(audio: np.ndarray, sample_rate: int, out_png: Path) -> Path:
    """Plot waveform and save to a PNG file.

    The plot contains a single trace with labeled axes.
    """
    _validate_audio_input(audio, sample_rate)

    _ensure_parent_dir(out_png)

    duration = len(audio) / float(sample_rate)
    t = np.linspace(0.0, duration, num=len(audio), endpoint=False)

    plt.figure(figsize=(10, 3))
    plt.plot(t, audio, color="#1f77b4", linewidth=1.0)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(str(out_png))
    plt.close()

    if not out_png.exists() or out_png.stat().st_size == 0:
        raise IOError(f"Failed to write waveform PNG at {out_png}")
    return out_png


