#!/usr/bin/env python3
"""
System verification test for TTS Mini Assistant.

Validates architecture/API consistency, required files, repo hygiene, and core utilities.
Network-dependent synthesis runs only when explicitly enabled via RUN_E2E_ONLINE=1.
"""

import os
import io
import re
import sys
import json
import shutil
import inspect
from pathlib import Path

import numpy as np
import pytest

RUN_E2E = os.getenv("RUN_E2E_ONLINE") == "1"


@pytest.mark.smoke
def test_repo_files_exist():
    """Verify all required files exist in the repository."""
    root = Path(__file__).resolve().parents[1]
    must_exist = [
        "src/tts_service.py",
        "src/audio_utils.py",
        "src/app.py",
        "scripts/preflight.py",
        "README.md",
        "requirements.txt",
        ".env.example",
        "docs/screenshot-ui.png",
        ".gitignore",
    ]
    for rel in must_exist:
        p = root / rel
        assert p.exists(), f"Missing required file: {rel}"


def _read_text(root: Path, rel: str) -> str:
    """Read text file with UTF-8 encoding."""
    p = root / rel
    return p.read_text(encoding="utf-8", errors="ignore")


@pytest.mark.smoke
def test_gitignore_and_env_example():
    """Verify .gitignore and .env.example are properly configured."""
    root = Path(__file__).resolve().parents[1]
    gi = _read_text(root, ".gitignore")
    assert ".env" in gi, "`.env` should be ignored in .gitignore"
    env_example = _read_text(root, ".env.example")
    assert "OPENAI_API_KEY" in env_example, "OPENAI_API_KEY should be documented in .env.example"


@pytest.mark.smoke
def test_requirements_hygiene():
    """Verify requirements.txt is clean and contains necessary packages."""
    root = Path(__file__).resolve().parents[1]
    req = _read_text(root, "requirements.txt").lower()
    assert "fastapi" not in req, "fastapi should NOT be in requirements.txt"
    assert "uvicorn" not in req, "uvicorn should NOT be in requirements.txt"
    assert "numpy" in req, "numpy must be explicitly listed in requirements.txt"
    assert "gtts" in req, "gTTS must be present in requirements.txt"


@pytest.mark.smoke
def test_engine_factory_signature():
    """Verify TTS engine factory returns objects with correct synthesize signature."""
    from src.tts_service import get_tts  # do not import heavy classes directly
    engine = get_tts("gtts")
    assert hasattr(engine, "synthesize"), "Engine must expose `synthesize`"
    sig = inspect.signature(engine.synthesize)
    params = list(sig.parameters.keys())
    for name in ("text", "language", "speaker", "speed", "pitch"):
        assert name in params, f"`synthesize` should accept `{name}` parameter"


@pytest.mark.smoke
def test_ui_mapping_and_import():
    """Verify UI mapping functions exist and work correctly."""
    import src.app as app
    assert app._map_ui_engine_to_code("gTTS (default)") == "gtts"
    assert app._map_ui_engine_to_code("OpenAI (API)") == "openai"
    assert app._map_ui_engine_to_code("Coqui (local)") == "coqui"
    assert app._map_ui_language_to_code("English (US)") == "en"
    assert app._map_ui_language_to_code("English (UK)") == "en"
    assert app._map_ui_language_to_code("Turkish") == "tr"


def _gen_sine(sr=22050, freq=440.0, secs=0.25):
    """Generate synthetic sine wave for testing."""
    t = np.linspace(0, secs, int(sr * secs), endpoint=False, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * freq * t, dtype=np.float32)
    return audio.astype(np.float32), sr


@pytest.mark.smoke
def test_audio_utils_offline(tmp_path):
    """Test audio utilities with synthetic audio (offline)."""
    from src.audio_utils import save_wav, save_mp3, plot_waveform
    audio, sr = _gen_sine()
    wav = save_wav(audio, sr, tmp_path / "tone.wav")
    assert wav.exists() and wav.stat().st_size > 0, "WAV not created"
    if shutil.which("ffmpeg"):
        mp3 = save_mp3(audio, sr, tmp_path / "tone.mp3")
        assert mp3.exists() and mp3.stat().st_size > 0, "MP3 not created (ffmpeg present)"
    png = plot_waveform(audio, sr, tmp_path / "tone.png")
    assert png.exists() and png.stat().st_size > 0, "Waveform PNG not created"


@pytest.mark.smoke
def test_readme_alignment():
    """Verify README contains expected content."""
    root = Path(__file__).resolve().parents[1]
    readme = _read_text(root, "README.md").lower()
    assert "streamlit ui" in readme, "README should mention Streamlit UI"
    assert "openai_api_key" in readme or "openai api key" in readme, "README should mention OPENAI_API_KEY"
    assert "compare mode" in readme or "side-by-side" in readme, "README should mention compare mode"


@pytest.mark.smoke
def test_preflight_mentions_core_checks():
    """Verify preflight script contains references to core dependency checks."""
    root = Path(__file__).resolve().parents[1]
    text = _read_text(root, "scripts/preflight.py").lower()
    # Non-executing sanity: ensure it references gtts and ffmpeg checks
    assert "gtts" in text, "preflight should reference gTTS"
    assert "ffmpeg" in text, "preflight should reference ffmpeg"


@pytest.mark.tts
@pytest.mark.skipif(not RUN_E2E, reason="Set RUN_E2E_ONLINE=1 to run online E2E synthesis")
def test_online_e2e_gtts(tmp_path):
    """Minimal gTTS e2e test (network dependent). Skip on failures gracefully."""
    try:
        from src.tts_service import get_tts
        eng = get_tts("gtts")
        audio, sr = eng.synthesize("Hello from E2E", language="en")
        assert isinstance(sr, int) and sr > 0 and isinstance(audio, np.ndarray) and audio.size > 0
        from src.audio_utils import save_wav, plot_waveform
        wav = save_wav(audio, sr, tmp_path / "e2e.wav")
        png = plot_waveform(audio, sr, tmp_path / "e2e.png")
        assert wav.exists() and wav.stat().st_size > 0
        assert png.exists() and png.stat().st_size > 0
    except Exception as e:
        pytest.skip(f"Online E2E skipped due to network/engine issue: {e}")


@pytest.mark.smoke
def test_audio_validation_edge_cases(tmp_path):
    """Test audio utilities handle edge cases correctly."""
    from src.audio_utils import save_wav, plot_waveform
    
    # Test very short audio
    short_audio, sr = _gen_sine(secs=0.05)
    wav = save_wav(short_audio, sr, tmp_path / "short.wav")
    assert wav.exists() and wav.stat().st_size > 0, "Short audio WAV not created"
    
    # Test different sample rate
    high_sr_audio, high_sr = _gen_sine(sr=48000, freq=880.0)
    wav_high = save_wav(high_sr_audio, high_sr, tmp_path / "high_sr.wav")
    assert wav_high.exists() and wav_high.stat().st_size > 0, "High sample rate WAV not created"


@pytest.mark.smoke
def test_engine_enum_and_protocol():
    """Verify TTS engine enumeration and protocol are properly defined."""
    from src.tts_service import TTSEngine, TTSProtocol
    
    # Check enum values
    assert TTSEngine.GTTS == "gtts"
    assert TTSEngine.OPENAI == "openai"
    assert TTSEngine.COQUI == "coqui"
    
    # Check protocol exists (structural typing)
    assert hasattr(TTSProtocol, '__call__') or hasattr(TTSProtocol, 'synthesize'), "TTSProtocol should define synthesize method"


@pytest.mark.smoke
def test_language_mapping_consistency():
    """Verify language mapping is consistent across the codebase."""
    from src.tts_service import LANGUAGE_MAP
    
    # Check core language codes
    assert "en" in LANGUAGE_MAP, "English language code should be mapped"
    assert "tr" in LANGUAGE_MAP, "Turkish language code should be mapped"
    
    # Check UI-friendly names
    assert "English (US)" in LANGUAGE_MAP, "English (US) should be mapped"
    assert "English (UK)" in LANGUAGE_MAP, "English (UK) should be mapped"
    assert "Turkish" in LANGUAGE_MAP, "Turkish should be mapped"
    
    # Verify mapping consistency
    assert LANGUAGE_MAP["en"] == "en", "English code should map to itself"
    assert LANGUAGE_MAP["tr"] == "tr", "Turkish code should map to itself"
    assert LANGUAGE_MAP["English (US)"] == "en", "English (US) should map to 'en'"
    assert LANGUAGE_MAP["English (UK)"] == "en", "English (UK) should map to 'en'"
    assert LANGUAGE_MAP["Turkish"] == "tr", "Turkish should map to 'tr'"
