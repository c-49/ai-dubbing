"""Stage 1: extract audio from an episode's source video.

Usage: python src/extract.py work/vincenzo/test_clip
Reads:  <episode_dir>/source.mp4
Writes: <episode_dir>/audio.wav  (44.1kHz stereo PCM, for Demucs next)
"""
import subprocess
import sys
from pathlib import Path

import pathfix  # noqa: F401  (patches PATH for ffmpeg)


def extract_audio(episode_dir: Path) -> Path:
    source = episode_dir / "source.mp4"
    audio_out = episode_dir / "audio.wav"
    if not source.exists():
        raise FileNotFoundError(f"no source video at {source}")
    if audio_out.exists():
        return audio_out

    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            str(audio_out),
        ],
        check=True,
    )
    return audio_out


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    out = extract_audio(episode_dir)
    print(f"wrote {out}")
