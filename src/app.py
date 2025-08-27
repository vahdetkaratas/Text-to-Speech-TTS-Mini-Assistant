#!/usr/bin/env python3
"""
Streamlit UI for TTS Mini Assistant.

Provides a clean interface for text-to-speech synthesis with multiple engines,
compare mode, and downloadable audio files.
"""

import os
import sys
import tempfile
import atexit
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

# Add parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np

from src.tts_service import get_tts
from src.audio_utils import save_wav, save_mp3, plot_waveform, PYDUB_AVAILABLE

if TYPE_CHECKING:
    from src.tts_service import TTSEngine


# UI Mapping Functions
def _map_ui_engine_to_code(label: str) -> str:
    """Map UI engine labels to internal engine codes."""
    mapping = {
        "gTTS (default)": "gtts",
        "OpenAI (API)": "openai",
        "Coqui (local)": "coqui",
    }
    return mapping.get(label, "gtts")


def _map_ui_language_to_code(label: str) -> str:
    """Map UI language labels to internal language codes."""
    mapping = {
        "English (US)": "en",
        "English (UK)": "en",  # Note: gTTS doesn't support UK accent
        "Turkish": "tr",
    }
    return mapping.get(label, "en")


def _get_available_engines() -> list[str]:
    """Get list of available engines based on current environment."""
    engines = ["gTTS (default)"]
    
    # Check if OpenAI is available
    try:
        import openai
        if os.getenv("OPENAI_API_KEY"):
            engines.append("OpenAI (API)")
    except ImportError:
        pass
    
    # Check if Coqui is available (and Python version compatible)
    try:
        import sys
        if sys.version_info < (3, 13):
            import TTS
            engines.append("Coqui (local)")
    except ImportError:
        pass
    
    return engines


def _synthesize_with_engine(
    engine_label: str, 
    text: str, 
    language: str, 
    speed: float = 1.0, 
    pitch: float = 0.0
) -> Optional[Tuple[np.ndarray, int]]:
    """Synthesize text with specified engine, handling errors gracefully."""
    try:
        engine_code = _map_ui_engine_to_code(engine_label)
        lang_code = _map_ui_language_to_code(language)
        
        if engine_code == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OPENAI_API_KEY environment variable not set")
                return None
            
            svc = get_tts(engine="openai", default_lang=lang_code)
            audio, sr = svc.synthesize(text, language=lang_code, speed=speed, pitch=pitch)
            return audio, sr
        
        elif engine_code == "coqui":
            try:
                svc = get_tts(engine="coqui", default_lang=lang_code)
                audio, sr = svc.synthesize(text, language=lang_code, speed=speed, pitch=pitch)
                return audio, sr
            except RuntimeError as e:
                if "not supported on Python 3.13" in str(e):
                    st.warning("⚠️ Coqui TTS is not supported on Python 3.13+. Skipping...")
                else:
                    st.error(f"❌ Coqui TTS error: {e}")
                return None
        
        else:  # gTTS
            svc = get_tts(engine="gtts", default_lang=lang_code)
            audio, sr = svc.synthesize(text, language=lang_code, speed=speed, pitch=pitch)
            return audio, sr
            
    except Exception as e:
        st.error(f"❌ Synthesis failed with {engine_label}: {e}")
        return None


def _create_audio_artifacts(audio: np.ndarray, sr: int, temp_dir: Path) -> dict:
    """Create audio artifacts (WAV, MP3, waveform) in temp directory."""
    artifacts = {}
    
    # Save WAV
    wav_path = temp_dir / "audio.wav"
    save_wav(audio, sr, wav_path)
    artifacts["wav"] = wav_path
    
    # Save MP3 (if available)
    if PYDUB_AVAILABLE:
        try:
            mp3_path = temp_dir / "audio.mp3"
            save_mp3(audio, sr, mp3_path)
            artifacts["mp3"] = mp3_path
        except Exception as e:
            st.warning(f"⚠️ MP3 export failed: {e}")
    else:
        st.info("ℹ️ MP3 export not available (pydub compatibility issue)")
    
    # Create waveform
    png_path = temp_dir / "waveform.png"
    plot_waveform(audio, sr, png_path)
    artifacts["waveform"] = png_path
    
    return artifacts


def _render_result_block(
    engine_label: str, 
    audio: np.ndarray, 
    sr: int, 
    artifacts: dict,
    col: Optional[st.container] = None
):
    """Render a single result block with audio player, downloads, and waveform."""
    container = col if col else st
    
    container.subheader(f"{engine_label} Result")
    
    # Audio player
    with open(artifacts["wav"], "rb") as f:
        container.audio(f.read(), format="audio/wav")
    
    # Download buttons
    col1, col2 = container.columns(2)
    
    with col1:
        with open(artifacts["wav"], "rb") as f:
            st.download_button(
                label="📥 Download WAV",
                data=f.read(),
                file_name="tts_output.wav",
                mime="audio/wav"
            )
    
    with col2:
        if "mp3" in artifacts:
            with open(artifacts["mp3"], "rb") as f:
                st.download_button(
                    label="📥 Download MP3",
                    data=f.read(),
                    file_name="tts_output.mp3",
                    mime="audio/mp3"
                )
        else:
            st.button("📥 Download MP3", disabled=True, help="MP3 export not available")
    
    # Waveform
    container.image(artifacts["waveform"], caption="Audio Waveform", use_column_width=True)


def _cleanup_temp_files():
    """Clean up temporary files from session state."""
    if "temp_dirs" in st.session_state:
        for temp_dir in st.session_state.temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        st.session_state.temp_dirs.clear()


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="TTS Mini Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    # Initialize temp file tracking
    if "temp_dirs" not in st.session_state:
        st.session_state.temp_dirs = []
    
    # Register cleanup on exit
    atexit.register(_cleanup_temp_files)
    
    st.title("🎤 Text-to-Speech Mini Assistant")
    st.markdown("Generate speech from text using multiple TTS engines")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Language selection
        language = st.selectbox(
            "Language",
            ["English (US)", "English (UK)", "Turkish"],
            help="Note: gTTS doesn't support true UK accent differences"
        )
        
        # Engine selection
        available_engines = _get_available_engines()
        primary_engine = st.selectbox(
            "Primary Engine",
            available_engines,
            help="Select the main TTS engine to use"
        )
        
        # Compare mode
        compare_mode = st.checkbox(
            "Compare Mode",
            help="Compare two engines side-by-side"
        )
        
        secondary_engine = None
        if compare_mode:
            secondary_engine = st.selectbox(
                "Secondary Engine",
                available_engines,
                help="Select a second engine for comparison"
            )
        
        st.divider()
        
        # Speed and pitch controls
        st.subheader("Audio Controls")
        speed = st.slider("Speed", 0.5, 1.5, 1.0, 0.1, help="Speech speed multiplier")
        pitch = st.slider("Pitch", -5.0, 5.0, 0.0, 0.5, help="Pitch adjustment")
        
        # Engine-specific notes
        if "gTTS" in primary_engine or (secondary_engine and "gTTS" in secondary_engine):
            st.caption("⚠️ Speed/Pitch may be ignored by gTTS")
        
        if "OpenAI" in primary_engine or (secondary_engine and "OpenAI" in secondary_engine):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("❌ OPENAI_API_KEY not set")
    
    # Main content area
    st.header("Text Input")
    
    # Example texts
    examples = {
        "English (US)": "Hello! This is a test of the text-to-speech system. How does it sound?",
        "English (UK)": "Hello! This is a test of the text-to-speech system. How does it sound?",
        "Turkish": "Merhaba! Bu metin-konuşma sisteminin bir testidir. Nasıl ses veriyor?"
    }
    
    # Text input with placeholder
    text = st.text_area(
        "Enter text to synthesize",
        value=examples.get(language, "Enter your text here..."),
        height=150,
        help="Enter the text you want to convert to speech"
    )
    
    # Generate button
    if st.button("🎤 Generate Speech", type="primary", use_container_width=True):
        if not text.strip():
            st.error("❌ Please enter some text to synthesize")
            return
        
        # Show processing message
        with st.spinner("Generating speech..."):
            # Create temporary directory for artifacts
            temp_dir = tempfile.mkdtemp()
            temp_path = Path(temp_dir)
            st.session_state.temp_dirs.append(temp_dir)
            
            # Generate with primary engine
            primary_result = _synthesize_with_engine(
                primary_engine, text, language, speed, pitch
            )
                
            if primary_result is None:
                st.error("❌ Primary engine failed to generate speech")
                return
            
            primary_audio, primary_sr = primary_result
            primary_artifacts = _create_audio_artifacts(primary_audio, primary_sr, temp_path)
            
            # Handle compare mode
            if compare_mode and secondary_engine:
                secondary_result = _synthesize_with_engine(
                    secondary_engine, text, language, speed, pitch
                )
                
                if secondary_result is None:
                    st.warning("⚠️ Secondary engine failed, showing primary result only")
                    compare_mode = False
                else:
                    secondary_audio, secondary_sr = secondary_result
                    secondary_artifacts = _create_audio_artifacts(secondary_audio, secondary_sr, temp_path)
            
            # Display results
            st.success("✅ Speech generated successfully!")
            
            if compare_mode and secondary_engine:
                # Two-column layout for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    _render_result_block(primary_engine, primary_audio, primary_sr, primary_artifacts, col1)
                
                with col2:
                    _render_result_block(secondary_engine, secondary_audio, secondary_sr, secondary_artifacts, col2)
            else:
                # Single result
                _render_result_block(primary_engine, primary_audio, primary_sr, primary_artifacts)
    
    # Footer
    st.divider()
    st.caption(
        "💡 **Tips:** "
        "• gTTS requires internet connection • "
        "• Coqui may not work on Python 3.13+ • "
        "• MP3 export requires ffmpeg • "
        "• OpenAI requires API key"
    )


if __name__ == "__main__":
    main()
