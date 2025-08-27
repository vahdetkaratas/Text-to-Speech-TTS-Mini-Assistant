import os
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tts_service import get_tts, TTSEngine
from src.audio_utils import save_wav


RUN_TTS = os.getenv("RUN_TTS_TEST") == "1"


@pytest.mark.skipif(not RUN_TTS, reason="Set RUN_TTS_TEST=1 to run TTS smoke test")
@pytest.mark.tts
def test_gtts_smoke(tmp_path):
    """Test gTTS engine (default, works on Python 3.13+)."""
    svc = get_tts(engine="gtts", default_lang="en")
    audio, sr = svc.synthesize("Hello from TTS", language="en")

    assert isinstance(audio, np.ndarray)
    assert isinstance(sr, int) and sr > 0
    assert audio.ndim == 1 and audio.dtype == np.float32

    out_wav = tmp_path / "gtts.wav"
    written = save_wav(audio, sr, out_wav)
    assert written.exists() and written.stat().st_size > 0


@pytest.mark.skipif(not RUN_TTS, reason="Set RUN_TTS_TEST=1 to run TTS smoke test")
@pytest.mark.skipif(sys.version_info >= (3, 13), reason="Coqui not supported on Python 3.13+")
@pytest.mark.tts
def test_coqui_tts_smoke(tmp_path):
    """Test Coqui TTS engine (optional, not available on Python 3.13+)."""
    try:
        svc = get_tts(engine="coqui", default_lang="en")
        audio, sr = svc.synthesize("Hello from TTS", language="en")

        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int) and sr > 0
        assert audio.ndim == 1 and audio.dtype == np.float32

        out_wav = tmp_path / "coqui.wav"
        written = save_wav(audio, sr, out_wav)
        assert written.exists() and written.stat().st_size > 0
    except RuntimeError as e:
        if "not available" in str(e):
            pytest.skip(f"Coqui TTS not available: {e}")
        else:
            raise


@pytest.mark.skipif(not RUN_TTS, reason="Set RUN_TTS_TEST=1 to run TTS smoke test")
@pytest.mark.tts
def test_openai_tts_smoke(tmp_path):
    """Test OpenAI TTS engine (optional, requires API key)."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    try:
        svc = get_tts(engine="openai", default_lang="en")
        audio, sr = svc.synthesize("Hello from TTS", language="en")

        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int) and sr > 0
        assert audio.ndim == 1 and audio.dtype == np.float32

        out_wav = tmp_path / "openai.wav"
        written = save_wav(audio, sr, out_wav)
        assert written.exists() and written.stat().st_size > 0
    except RuntimeError as e:
        if "not available" in str(e) or "not set" in str(e):
            pytest.skip(f"OpenAI TTS not available: {e}")
        else:
            raise


@pytest.mark.tts
def test_empty_text_validation():
    """Test that all engines properly validate empty text input."""
    engines_to_test = ["gtts"]
    
    # Add OpenAI if API key is available
    if os.getenv("OPENAI_API_KEY"):
        engines_to_test.append("openai")
    
    # Add Coqui if Python < 3.13 and package available
    if sys.version_info < (3, 13):
        try:
            import TTS
            engines_to_test.append("coqui")
        except ImportError:
            pass
    
    for engine in engines_to_test:
        try:
            svc = get_tts(engine=engine, default_lang="en")
            
            # Test empty string
            with pytest.raises(ValueError, match="must be a non-empty string"):
                svc.synthesize("", language="en")
            
            # Test whitespace-only string
            with pytest.raises(ValueError, match="must be a non-empty string"):
                svc.synthesize("   ", language="en")
                
        except RuntimeError as e:
            if "not available" in str(e) or "not set" in str(e):
                pytest.skip(f"{engine} not available: {e}")
            else:
                raise


@pytest.mark.skipif(not RUN_TTS, reason="Set RUN_TTS_TEST=1 to run TTS smoke test")
@pytest.mark.tts
def test_long_text_handling(tmp_path):
    """Test handling of longer text input (>1000 chars)."""
    # Create a long but reasonable text (avoid timeouts)
    long_text = "This is a test of long text synthesis. " * 30  # ~1200 chars
    
    engines_to_test = ["gtts"]
    
    # Add OpenAI if API key is available
    if os.getenv("OPENAI_API_KEY"):
        engines_to_test.append("openai")
    
    # Add Coqui if Python < 3.13 and package available
    if sys.version_info < (3, 13):
        try:
            import TTS
            engines_to_test.append("coqui")
        except ImportError:
            pass
    
    for engine in engines_to_test:
        try:
            svc = get_tts(engine=engine, default_lang="en")
            
            # Test long text synthesis
            audio, sr = svc.synthesize(long_text, language="en")
            
            # Basic validation
            assert isinstance(audio, np.ndarray)
            assert isinstance(sr, int) and sr > 0
            assert audio.ndim == 1 and audio.dtype == np.float32
            assert len(audio) > 0, "Long text should produce audio"
            
            # Save to verify it works
            out_wav = tmp_path / f"long_text_{engine}.wav"
            written = save_wav(audio, sr, out_wav)
            assert written.exists() and written.stat().st_size > 0
                
        except RuntimeError as e:
            if "not available" in str(e) or "not set" in str(e):
                pytest.skip(f"{engine} not available: {e}")
            elif "rate limit" in str(e).lower() or "timeout" in str(e).lower():
                pytest.skip(f"{engine} rate limited or timed out: {e}")
            else:
                raise


@pytest.mark.skipif(not RUN_TTS, reason="Set RUN_TTS_TEST=1 to run TTS smoke test")
@pytest.mark.tts
def test_unicode_turkish_diacritics(tmp_path):
    """Test Unicode text with Turkish diacritics."""
    # Turkish text with diacritics
    turkish_text = "Merhaba dünya, çığ köprü şüphe örneği."
    
    # Test with gTTS (most reliable for Unicode)
    try:
        svc = get_tts(engine="gtts", default_lang="tr")
        
        # Test Turkish text synthesis
        audio, sr = svc.synthesize(turkish_text, language="tr")
        
        # Basic validation
        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int) and sr > 0
        assert audio.ndim == 1 and audio.dtype == np.float32
        assert len(audio) > 0, "Turkish text should produce audio"
        
        # Save to verify it works
        out_wav = tmp_path / "turkish_unicode.wav"
        written = save_wav(audio, sr, out_wav)
        assert written.exists() and written.stat().st_size > 0
        
    except RuntimeError as e:
        if "not available" in str(e) or "network" in str(e).lower():
            pytest.skip(f"gTTS not available or network issue: {e}")
        else:
            raise


