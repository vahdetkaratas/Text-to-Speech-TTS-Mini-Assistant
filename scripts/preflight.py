#!/usr/bin/env python3
"""Preflight checks for TTS Mini Assistant environment."""

import sys
import shutil
from pathlib import Path


def check_python_version() -> None:
    """Check Python version >= 3.10."""
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 10):
        print("❌ Python 3.10+ required")
        sys.exit(1)
    print("✅ Python version OK")


def check_imports() -> None:
    """Check required package imports."""
    # Core required packages
    required_packages = [
        ("soundfile", "soundfile"),
        ("matplotlib", "matplotlib"),
        ("numpy", "numpy"),
        ("pytest", "pytest"),
        ("streamlit", "streamlit"),
        ("gTTS", "gtts"),
    ]
    
    failed_imports = []
    
    # Check pydub separately due to Python 3.13 compatibility issues
    try:
        import pydub
        print("✅ pydub")
    except ImportError as e:
        if "pyaudioop" in str(e):
            print("⚠️  pydub - MP3 export disabled (Python 3.13 compatibility)")
        else:
            print("❌ pydub - not installed")
            failed_imports.append("pydub")
    
    # Check core packages
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - not installed")
            failed_imports.append(package_name)
    
    # Check optional packages
    print("\n📦 Optional packages:")
    
    # Check Coqui TTS
    try:
        import TTS
        if sys.version_info >= (3, 13):
            print("⚠️  coqui-tts - available but not supported on Python 3.13+")
        else:
            print("✅ coqui-tts")
    except ImportError:
        if sys.version_info >= (3, 13):
            print("⚠️  coqui-tts - optional and not supported on Python 3.13+")
        else:
            print("⚠️  coqui-tts - optional (not installed)")
    
    # Check OpenAI
    try:
        import openai
        print("✅ openai")
    except ImportError:
        print("⚠️  openai - optional (not installed)")
    
    if failed_imports:
        print(f"\n❌ Missing core packages: {', '.join(failed_imports)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def check_ffmpeg() -> None:
    """Check ffmpeg availability."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ ffmpeg found at: {ffmpeg_path}")
    else:
        print("⚠️  ffmpeg not found - MP3 export will be skipped")
        print("Install ffmpeg for MP3 support:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: choco install ffmpeg")


def main() -> None:
    """Run all preflight checks."""
    print("🔍 TTS Mini Assistant - Preflight Check")
    print("=" * 40)
    
    check_python_version()
    print()
    
    print("📦 Checking package imports...")
    check_imports()
    print()
    
    print("🎵 Checking ffmpeg...")
    check_ffmpeg()
    print()
    
    print("✅ All checks passed! Environment is ready.")
    print("\nNext steps:")
    print("  pytest -q                    # Run basic tests")
    print("  RUN_TTS_TEST=1 pytest -q -k test_tts  # Run TTS smoke test")
    print("  streamlit run src/app.py     # Launch Streamlit UI")


if __name__ == "__main__":
    main()
