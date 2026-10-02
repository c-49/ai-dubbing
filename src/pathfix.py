"""
Makes the project self-contained: puts the bundled ffmpeg/eSpeak NG (under
<project>/bin) on PATH and the DLL search path, and points the Hugging Face
and torch caches at <project>/models, so the folder can be copied to another
machine or path with no edits. Import this before calling ffmpeg/espeak-ng
via subprocess or loading any model.

If a bundled tool is missing, whatever is already on the system PATH is used.
"""
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252, which can't print Korean/Japanese/etc.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Models travel with the project. setdefault so an explicit override wins.
_MODELS = PROJECT_ROOT / "models"
os.environ.setdefault("HF_HOME", str(_MODELS / "hf"))
os.environ.setdefault("TORCH_HOME", str(_MODELS / "torch"))
# The HF token is a per-user credential and must not live inside the
# (copyable) project folder, so keep reading it from the user's home.
os.environ.setdefault(
    "HF_TOKEN_PATH", str(Path.home() / ".cache" / "huggingface" / "token")
)

# ffmpeg's shared DLLs are needed by torchcodec (pyannote.audio 4.x), not just
# ffmpeg.exe, so bin/ffmpeg holds the shared build.
_EXTRA_PATHS = (
    PROJECT_ROOT / "bin" / "ffmpeg",
    PROJECT_ROOT / "bin" / "espeak-ng",
)

for _p in map(str, _EXTRA_PATHS):
    if not os.path.isdir(_p):
        continue
    if _p not in os.environ["PATH"]:
        os.environ["PATH"] = _p + os.pathsep + os.environ["PATH"]
    # Python 3.8+ on Windows no longer searches PATH for native DLL loading
    # (ctypes/torch native libs) -- only subprocess calls to .exe files still
    # use PATH. os.add_dll_directory is the documented fix for the former.
    try:
        os.add_dll_directory(_p)
    except (AttributeError, OSError):
        pass
