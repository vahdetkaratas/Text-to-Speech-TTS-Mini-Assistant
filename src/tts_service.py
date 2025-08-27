from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple, Protocol
import io

import numpy as np

# Engine enumeration
class TTSEngine(str, Enum):
    GTTS = "gtts"
    OPENAI = "openai"
    COQUI = "coqui"


# Protocol for TTS services
class TTSProtocol(Protocol):
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech and return (audio, sample_rate)."""
        ...


# Language mapping
LANGUAGE_MAP: Dict[str, str] = {
    "en": "en",
    "tr": "tr",
    "English (US)": "en",
    "English (UK)": "en",  # Note: gTTS doesn't support UK accent
    "Turkish": "tr",
}


@dataclass
class _LoadedModel:
    language: str
    tts: object


class GttsTTSService:
    """gTTS-based TTS service with lazy-loading and per-language cache.
    
    Uses Google Text-to-Speech for synthesis. Requires internet connection.
    
    Notes
    -----
    - Requires internet connection for synthesis
    - Speed and pitch parameters are ignored (gTTS limitation)
    - Speaker parameter is ignored (gTTS limitation)
    - Language mapping: "en" -> English, "tr" -> Turkish
    """
    
    def __init__(self, default_lang: str = "en") -> None:
        if default_lang not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported default language: {default_lang!r}. Supported: {list(LANGUAGE_MAP)}")
        self._default_lang: str = default_lang
        self._cache: Dict[str, _LoadedModel] = {}
    
    def set_language(self, lang: str) -> None:
        """Set the default language used by `synthesize` when `language` is None."""
        if lang not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported language: {lang!r}. Supported: {list(LANGUAGE_MAP)}")
        self._default_lang = lang
    
    def _ensure_model(self, lang: str) -> _LoadedModel:
        if lang in self._cache:
            return self._cache[lang]
        
        # Try to import gTTS
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError("gTTS TTS failed: Ensure 'gTTS' is installed. Run: pip install gTTS") from exc
        
        # Cache the gTTS class for efficiency
        loaded = _LoadedModel(language=lang, tts=gTTS)
        self._cache[lang] = loaded
        return loaded
    
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using gTTS.
        
        Returns (audio, sample_rate). The audio is a 1-D float32 numpy array in [-1, 1].
        
        Parameters
        ----------
        text: str
            Input text to synthesize.
        language: Optional[str]
            Language code ("en" or "tr"). Defaults to the service's default language.
        speaker: Optional[str]
            Ignored (gTTS limitation).
        speed: float
            Ignored (gTTS limitation).
        pitch: Optional[float]
            Ignored (gTTS limitation).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        
        lang = (language or self._default_lang).lower()
        model = self._ensure_model(lang)
        
        try:
            # gTTS requires internet connection
            # Create gTTS instance with text and language
            tts = model.tts(text=text, lang=lang)
            
            # Get audio bytes
            audio_bytes_io = io.BytesIO()
            tts.write_to_fp(audio_bytes_io)
            audio_bytes = audio_bytes_io.getvalue()
            
            # Convert MP3 bytes to waveform using pydub
            try:
                from pydub import AudioSegment
                
                # Load MP3 from bytes
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                
                # Convert to mono if stereo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Export as WAV bytes
                wav_bytes = io.BytesIO()
                audio_segment.export(wav_bytes, format="wav")
                wav_bytes.seek(0)
                
                # Read with soundfile
                import soundfile as sf
                audio_data, sample_rate = sf.read(wav_bytes)
                
            except ImportError:
                # Fallback: try direct soundfile read (may not work for MP3)
                import soundfile as sf
                audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
                
        except Exception as exc:
            raise RuntimeError("gTTS TTS failed: Check your internet connection") from exc
        
        audio = np.asarray(audio_data, dtype=np.float32).flatten()
        # Clamp to [-1, 1] just in case
        np.clip(audio, -1.0, 1.0, out=audio)
        return audio, int(sample_rate)


class CoquiTTSService:
    """Coqui TTS service (optional, may not work on Python 3.13+).
    
    Notes
    -----
    - May not be available on Python 3.13+ due to compatibility issues
    - Requires model downloads on first use
    - Supports speed/pitch control (when available)
    """
    
    def __init__(self, default_lang: str = "en") -> None:
        # Check Python version compatibility
        if sys.version_info >= (3, 13):
            raise RuntimeError("Coqui TTS failed: Not supported on Python 3.13+. Use gTTS or OpenAI engines instead")
        
        # Check if coqui-tts is available
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("Coqui TTS failed: Not available. Install with: pip install coqui-tts")
        
        self._default_lang = default_lang
        self._cache: Dict[str, _LoadedModel] = {}
        
        # Model mapping for Coqui
        self._model_map = {
            "en": "tts_models/en/ljspeech/tacotron2-DDC",
            "tr": "tts_models/tr/common-voice/glow-tts",
        }
    
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using Coqui TTS.
        
        Returns (audio, sample_rate). The audio is a 1-D float32 numpy array in [-1, 1].
        
        Parameters
        ----------
        text: str
            Input text to synthesize.
        language: Optional[str]
            Language code ("en" or "tr"). Defaults to the service's default language.
        speaker: Optional[str]
            Speaker ID (model-dependent, may be ignored).
        speed: float
            Speech speed multiplier (model-dependent).
        pitch: Optional[float]
            Pitch adjustment (model-dependent, may be ignored).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        
        # Check Python version compatibility
        if sys.version_info >= (3, 13):
            raise RuntimeError("Coqui TTS failed: Not supported on Python 3.13+. Use gTTS or OpenAI engines instead")
        
        lang = (language or self._default_lang).lower()
        if lang not in self._model_map:
            raise ValueError(f"Unsupported language: {lang!r}. Supported: {list(self._model_map)}")
        
        try:
            from TTS.api import TTS
        except ImportError:
            raise RuntimeError("Coqui TTS failed: Not available. Install with: pip install coqui-tts")
        
        try:
            # Initialize TTS with the appropriate model
            model_name = self._model_map[lang]
            tts = TTS(model_name)
            
            # Generate speech
            # Note: Coqui TTS parameters vary by model, so we use defaults
            wav = tts.tts(text=text)
            
            # Convert to numpy array
            audio = np.asarray(wav, dtype=np.float32)
            
            # Ensure mono and correct shape
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Normalize to [-1, 1] range
            if audio.max() > 0:
                audio = audio / audio.max()
            
            # Clamp to [-1, 1] just in case
            np.clip(audio, -1.0, 1.0, out=audio)
            
            # Coqui typically uses 22050 Hz sample rate
            # Note: This is an assumption; actual sample rate may differ per model
            sample_rate = 22050
            
            return audio, sample_rate
            
        except Exception as exc:
            raise RuntimeError("Coqui TTS failed: Check your internet connection and try again")


class OpenAITTSService:
    """OpenAI TTS service (optional, requires API key).
    
    Notes
    -----
    - Requires OPENAI_API_KEY environment variable
    - Speed and pitch parameters are ignored (OpenAI API limitation)
    - Speaker parameter is ignored (OpenAI API limitation)
    - Language mapping: "en" -> English, "tr" -> Turkish
    """
    
    def __init__(self, default_lang: str = "en") -> None:
        if default_lang not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported default language: {default_lang!r}. Supported: {list(LANGUAGE_MAP)}")
        self._default_lang: str = default_lang
    
    def set_language(self, lang: str) -> None:
        """Set the default language used by `synthesize` when `language` is None."""
        if lang not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported language: {lang!r}. Supported: {list(LANGUAGE_MAP)}")
        self._default_lang = lang
    
    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        speed: float = 1.0,
        pitch: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech using OpenAI TTS.
        
        Returns (audio, sample_rate). The audio is a 1-D float32 numpy array in [-1, 1].
        
        Parameters
        ----------
        text: str
            Input text to synthesize.
        language: Optional[str]
            Language code ("en" or "tr"). Defaults to the service's default language.
        speaker: Optional[str]
            Ignored (OpenAI API limitation).
        speed: float
            Ignored (OpenAI API limitation).
        pitch: Optional[float]
            Ignored (OpenAI API limitation).
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        
        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI TTS failed: OPENAI_API_KEY environment variable not set")
        
        try:
            import openai
        except ImportError:
            raise RuntimeError("OpenAI TTS failed: Not available. Install with: pip install openai")
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text,
            )
            
            # Convert response to numpy array
            audio_bytes = response.content
            import soundfile as sf
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            
            audio = np.asarray(audio_data, dtype=np.float32).flatten()
            np.clip(audio, -1.0, 1.0, out=audio)
            return audio, int(sample_rate)
            
        except Exception as exc:
            raise RuntimeError("OpenAI TTS failed: Check your API key and network connectivity")





def get_tts(engine: str = "gtts", default_lang: str = "en") -> TTSProtocol:
    """Factory function to get TTS service.
    
    Parameters
    ----------
    engine: str
        Engine to use: "gtts", "openai", or "coqui"
    default_lang: str
        Default language for synthesis
        
    Returns
    -------
    TTSProtocol
        TTS service instance with synthesize() method
    """
    engine = engine.lower()
    
    if engine == TTSEngine.GTTS:
        return GttsTTSService(default_lang=default_lang)
    
    elif engine == TTSEngine.OPENAI:
        return OpenAITTSService(default_lang=default_lang)
    
    elif engine == TTSEngine.COQUI:
        try:
            return CoquiTTSService(default_lang=default_lang)
        except RuntimeError as e:
            raise RuntimeError(f"Coqui TTS failed: {e}")
    
    else:
        raise ValueError(f"Unknown engine: {engine}. Supported: {[e.value for e in TTSEngine]}")


# Note: Speed and pitch parameters may be ignored by some engines (e.g., gTTS, OpenAI)
# and are model-dependent for others (e.g., Coqui). Check engine documentation for details.


