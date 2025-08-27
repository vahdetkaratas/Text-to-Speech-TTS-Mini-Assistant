import numpy as np
from pathlib import Path

import shutil
import pytest

from src.audio_utils import save_wav, save_mp3, plot_waveform, PYDUB_AVAILABLE


def gen_sine(sr: int = 22050, freq: float = 440.0, secs: float = 1.0) -> tuple[np.ndarray, int]:
    t = np.linspace(0.0, secs, int(sr * secs), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return audio, sr


@pytest.mark.skipif(not PYDUB_AVAILABLE, reason="pydub not available (Python 3.13 compatibility)")
def test_audio_utils_end_to_end(tmp_path):
    audio, sr = gen_sine()
    out_dir = tmp_path / "outs"

    wav_path = out_dir / "tone.wav"
    mp3_path = out_dir / "tone.mp3"
    png_path = out_dir / "tone.png"

    wav_written = save_wav(audio, sr, wav_path)
    assert wav_written.exists() and wav_written.stat().st_size > 0

    # Skip MP3 if ffmpeg not available
    if shutil.which("ffmpeg"):
        mp3_written = save_mp3(audio, sr, mp3_path)
        assert mp3_written.exists() and mp3_written.stat().st_size > 0
    else:
        pytest.skip("ffmpeg not available, skipping MP3 test")

    png_written = plot_waveform(audio, sr, png_path)
    assert png_written.exists() and png_written.stat().st_size > 0


@pytest.mark.smoke
def test_audio_utils_edge_cases(tmp_path):
    """Test edge cases: very short audio, different sample rates."""
    out_dir = tmp_path / "edge_cases"
    
    # Test very short audio (0.05 seconds)
    short_audio, sr = gen_sine(sr=22050, freq=440.0, secs=0.05)
    assert len(short_audio) > 0, "Short audio should have samples"
    
    wav_path = out_dir / "short.wav"
    png_path = out_dir / "short.png"
    
    wav_written = save_wav(short_audio, sr, wav_path)
    assert wav_written.exists() and wav_written.stat().st_size > 0
    
    png_written = plot_waveform(short_audio, sr, png_path)
    assert png_written.exists() and png_written.stat().st_size > 0
    
    # Test different sample rate
    high_sr_audio, high_sr = gen_sine(sr=48000, freq=880.0, secs=0.1)
    wav_path_high = out_dir / "high_sr.wav"
    
    wav_written_high = save_wav(high_sr_audio, high_sr, wav_path_high)
    assert wav_written_high.exists() and wav_written_high.stat().st_size > 0


@pytest.mark.smoke
def test_audio_round_trip(tmp_path):
    """Test audio fidelity when saving and loading WAV files."""
    from src.audio_utils import save_wav
    import soundfile as sf
    import numpy as np
    
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)

    wav = save_wav(audio, sr, tmp_path / "roundtrip.wav")
    loaded, sr_loaded = sf.read(wav, dtype="float32")

    assert sr_loaded == sr, "Sample rate mismatch on round-trip"
    assert loaded.shape[0] == audio.shape[0], "Length mismatch on round-trip"
    assert np.allclose(audio[:1000], loaded[:1000], atol=1e-4), "Audio not preserved on round-trip"


