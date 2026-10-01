"""Stage 2: separate vocals from music/effects with Demucs.

Usage: python src/separate.py work/vincenzo/test_clip
Reads:  <episode_dir>/audio.wav
Writes: <episode_dir>/vocals.wav, <episode_dir>/no_vocals.wav
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pathfix  # noqa: F401


def separate(episode_dir: Path) -> tuple[Path, Path]:
    audio_in = episode_dir / "audio.wav"
    if not audio_in.exists():
        raise FileNotFoundError(f"no audio at {audio_in}, run extract.py first")

    vocals_out = episode_dir / "vocals.wav"
    no_vocals_out = episode_dir / "no_vocals.wav"
    if vocals_out.exists() and no_vocals_out.exists():
        return vocals_out, no_vocals_out

    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems", "vocals",
            "--device", "cpu",
            "-o", str(episode_dir / "_demucs_out"),
            str(audio_in),
        ],
        check=True,
    )

    # demucs writes to <out>/htdemucs/<audio_stem>/{vocals,no_vocals}.wav
    stem_dir = episode_dir / "_demucs_out" / "htdemucs" / audio_in.stem
    shutil.copyfile(stem_dir / "vocals.wav", vocals_out)
    shutil.copyfile(stem_dir / "no_vocals.wav", no_vocals_out)
    shutil.rmtree(episode_dir / "_demucs_out")

    return vocals_out, no_vocals_out


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    vocals, no_vocals = separate(episode_dir)
    print(f"wrote {vocals}")
    print(f"wrote {no_vocals}")
