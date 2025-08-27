#!/usr/bin/env python3
"""Tests for Streamlit UI components."""

import pytest

from src.app import _map_ui_engine_to_code, _map_ui_language_to_code


@pytest.mark.smoke
def test_streamlit_app_imports():
    """Test that the Streamlit app can be imported without errors."""
    import src.app  # noqa: F401
    assert True


def test_engine_map():
    """Test engine label to code mapping."""
    assert _map_ui_engine_to_code("gTTS (default)") == "gtts"
    assert _map_ui_engine_to_code("OpenAI (API)") == "openai"
    assert _map_ui_engine_to_code("Coqui (local)") == "coqui"
    
    # Test fallback
    assert _map_ui_engine_to_code("unknown") == "gtts"


def test_lang_map():
    """Test language label to code mapping."""
    assert _map_ui_language_to_code("English (US)") == "en"
    assert _map_ui_language_to_code("English (UK)") == "en"
    assert _map_ui_language_to_code("Turkish") == "tr"
    
    # Test fallback
    assert _map_ui_language_to_code("unknown") == "en"


def test_mapping_consistency():
    """Test that mappings are consistent and complete."""
    # All UI labels should map to valid internal codes
    ui_engines = ["gTTS (default)", "OpenAI (API)", "Coqui (local)"]
    ui_languages = ["English (US)", "English (UK)", "Turkish"]
    
    for engine in ui_engines:
        code = _map_ui_engine_to_code(engine)
        assert code in ["gtts", "openai", "coqui"]
    
    for lang in ui_languages:
        code = _map_ui_language_to_code(lang)
        assert code in ["en", "tr"]
