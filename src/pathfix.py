"""
Adds ffmpeg/eSpeak-NG to PATH if a shell that predates their winget install
is still being used. New terminals already have them on PATH and this is a
no-op there. Import this before calling ffmpeg/espeak-ng via subprocess.
"""
import os
import sys

# Windows consoles default to cp1252, which can't print Korean/Japanese/etc.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

_EXTRA_PATHS = (
    r"C:\Users\reith\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin",
    # torchcodec (used by pyannote.audio 4.x) needs ffmpeg's shared DLLs, not
    # just the static ffmpeg.exe above.
    r"C:\Users\reith\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg.Shared_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build-shared\bin",
    r"C:\Program Files\eSpeak NG",
)

for _p in _EXTRA_PATHS:
    if _p not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + _p
    # Python 3.8+ on Windows no longer searches PATH for native DLL loading
    # (ctypes/torch native libs) -- only subprocess calls to .exe files still
    # use PATH. os.add_dll_directory is the documented fix for the former.
    if os.path.isdir(_p):
        try:
            os.add_dll_directory(_p)
        except (AttributeError, OSError):
            pass
