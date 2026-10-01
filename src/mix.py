"""Stage: mix English dub over music/effects (with ducking) and export mp4.

Usage: python src/mix.py work/vincenzo/test_clip
Reads:  <episode_dir>/segments.json, no_vocals.wav, vocals.wav, source.mp4
Writes: <episode_dir>/mixed_audio.wav, <episode_dir>/dubbed.mp4

Clips are placed at their segment start time (run timing.py first so they
fit their slots). Original vocals are ducked down only while a dub line is
actually playing, with a short crossfade -- not a flat reduction for the
whole episode -- then the final mix is loudness-normalized.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
MIX_SAMPLE_RATE = 44100


def load_stereo(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != target_sr:
        t = torchaudio.functional.resample(torch.from_numpy(audio.T), sr, target_sr)
        audio = t.numpy().T
    return audio


def build_dub_track(
    episode_dir: Path, segments: list[dict], total_samples: int, target_sr: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    track = np.zeros((total_samples, 2), dtype="float32")
    spans = []
    for seg in segments:
        if not seg.get("audio_path"):
            continue
        clip, sr = sf.read(episode_dir / seg["audio_path"], dtype="float32", always_2d=True)
        if clip.shape[1] == 1:
            clip = np.repeat(clip, 2, axis=1)
        if sr != target_sr:
            t = torchaudio.functional.resample(torch.from_numpy(clip.T), sr, target_sr)
            clip = t.numpy().T

        start_sample = round(seg["start"] * target_sr)
        if start_sample >= total_samples:
            continue
        end_sample = min(start_sample + len(clip), total_samples)
        track[start_sample:end_sample] += clip[: end_sample - start_sample]
        spans.append((start_sample, end_sample))
    return track, spans


def build_duck_envelope(
    total_samples: int, spans: list[tuple[int, int]], target_sr: int,
    ducked_level: float, unducked_level: float, fade_ms: float,
) -> np.ndarray:
    """1.0-scale envelope (reshaped for broadcasting) that dips to
    ducked_level while a dub clip plays and eases back to unducked_level
    elsewhere, with a linear crossfade of fade_ms at each edge."""
    envelope = np.full(total_samples, unducked_level, dtype="float32")
    fade_samples = max(1, round(fade_ms / 1000 * target_sr))
    for start, end in spans:
        envelope[start:end] = ducked_level
        fade_in_start = max(0, start - fade_samples)
        if fade_in_start < start:
            envelope[fade_in_start:start] = np.linspace(
                unducked_level, ducked_level, start - fade_in_start, dtype="float32"
            )
        fade_out_end = min(total_samples, end + fade_samples)
        if end < fade_out_end:
            envelope[end:fade_out_end] = np.linspace(
                ducked_level, unducked_level, fade_out_end - end, dtype="float32"
            )
    return envelope[:, None]  # (time, 1) for broadcasting over stereo


def normalize_loudness(in_path: Path, out_path: Path, sample_rate: int, target_lufs: float) -> None:
    # ffmpeg's loudnorm filter (EBU R128) handles gain + true-peak limiting
    # together -- a plain "scale to target LUFS" can demand peaks well past
    # 0 dBFS, and clamping those after the fact just undoes the gain again.
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(in_path),
            "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ar", str(sample_rate),
            str(out_path),
        ],
        check=True, capture_output=True,
    )


def mix(episode_dir: Path) -> Path:
    segments = json.loads((episode_dir / "segments.json").read_text(encoding="utf-8"))
    levels = yaml.safe_load(CONFIG_PATH.read_text())["mix_levels"]

    no_vocals = load_stereo(episode_dir / "no_vocals.wav", MIX_SAMPLE_RATE)
    vocals = load_stereo(episode_dir / "vocals.wav", MIX_SAMPLE_RATE)
    total_samples = len(no_vocals)

    dub_track, dub_spans = build_dub_track(episode_dir, segments, total_samples, MIX_SAMPLE_RATE)
    duck_envelope = build_duck_envelope(
        total_samples, dub_spans, MIX_SAMPLE_RATE,
        levels["original_vocals_ducked"], levels["original_vocals_unducked"], levels["duck_fade_ms"],
    )

    mixed = (
        no_vocals * levels["music_effects"]
        + vocals[:total_samples] * duck_envelope
        + dub_track * levels["english_dub"]
    )

    peak = float(np.max(np.abs(mixed)))
    if peak > 0.98:
        mixed *= 0.98 / peak

    prenorm_path = episode_dir / "_prenorm_audio.wav"
    sf.write(prenorm_path, mixed, MIX_SAMPLE_RATE)
    mixed_path = episode_dir / "mixed_audio.wav"
    normalize_loudness(prenorm_path, mixed_path, MIX_SAMPLE_RATE, levels["target_loudness_lufs"])
    prenorm_path.unlink()

    out_path = episode_dir / "dubbed.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(episode_dir / "source.mp4"),
            "-i", str(mixed_path),
            "-i", str(episode_dir / "audio.wav"),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-metadata:s:a:0", "title=English Dub",
            "-metadata:s:a:1", "title=Original Korean",
            "-disposition:a:0", "default",
            "-disposition:a:1", "0",
            str(out_path),
        ],
        check=True,
    )
    return out_path


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    out = mix(episode_dir)
    print(f"wrote {out}")
