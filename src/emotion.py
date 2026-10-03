"""Stage: estimate how each original line was delivered, and cut its audio
as a possible delivery reference for voice cloning.

Usage: python src/emotion.py work/<show>/<episode>
Reads:  <episode_dir>/segments.json (needs "speaker"), vocals.wav
Writes: <episode_dir>/line_refs/<id>.wav, and per segment in segments.json:
          "emotion":        "neutral" | "intense" | "soft"  (only if not already set)
          "line_reference": project-relative path of the line's own audio clip

The label is arousal only (how loud/high the line was compared with that
speaker's own usual level), measured from the separated vocals, so it works
for any source language with no extra model. It cannot tell anger from joy;
the reviewer can set richer labels (angry, sad, crying, whisper, ...) in the
review UI and those are never overwritten here.

How it is used: a cloning engine that can take a reference clip (OmniVoice)
is given the line's own clip instead of the speaker's usual reference
whenever the emotion isn't neutral, so the dub inherits that delivery. See
tts_engine.synthesize_segment.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf

import pathfix  # noqa: F401
import references

MIN_JUDGE_SECONDS = 1.0   # shorter fragments ("Sir.", "Okay.") are too little audio to judge
MIN_REF_SECONDS = 1.2     # shorter clips are too little for a model to pick up delivery from
MAX_REF_SECONDS = 10.0
MIN_VOICED_HZ, MAX_VOICED_HZ = 60.0, 400.0
INTENSE_SCORE = 1.5       # combined z-like score thresholds (see score())
SOFT_SCORE = -1.5
LOUD_DB_PER_POINT = 4.0   # +4 dB over the speaker's usual peak level = 1 point
SEMITONES_PER_POINT = 2.0 # +2 semitones over their usual pitch = 1 point


def line_features(audio: np.ndarray, sr: int) -> tuple[float, float | None]:
    """(loudness in dB, median F0 in Hz or None) for one line's mono audio."""
    frame = max(1, round(0.025 * sr))
    n = len(audio) // frame
    if n == 0:
        return -120.0, None
    frames = audio[: n * frame].reshape(n, frame)
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-8)
    loudness = float(np.percentile(rms_db, 90))   # peak-ish level: ignores pauses and tails
    f0 = None
    if len(audio) >= sr // 2:
        pitch = taf.detect_pitch_frequency(torch.from_numpy(audio), sr)
        voiced = pitch[(pitch > MIN_VOICED_HZ) & (pitch < MAX_VOICED_HZ)]
        if voiced.numel() >= 5:
            f0 = float(voiced.median())
    return loudness, f0


def score(loudness: float, f0: float | None, base_loudness: float, base_f0: float | None) -> float:
    s = (loudness - base_loudness) / LOUD_DB_PER_POINT
    if f0 and base_f0:
        s += 12 * float(np.log2(f0 / base_f0)) / SEMITONES_PER_POINT
    return s


def analyze(episode_dir: Path) -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    audio, sr = sf.read(episode_dir / "vocals.wav", dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)

    feats = {}
    for seg in segments:
        clip = mono[max(0, round(seg["start"] * sr)): round(seg["end"] * sr)]
        feats[seg["id"]] = line_features(clip, sr)

    def baseline(group):
        loud = [feats[s["id"]][0] for s in group]
        f0s = [feats[s["id"]][1] for s in group if feats[s["id"]][1]]
        return float(np.median(loud)), (float(np.median(f0s)) if f0s else None)

    overall = baseline(segments)
    by_speaker = {}
    for spk in {s.get("speaker") for s in segments}:
        group = [s for s in segments if s.get("speaker") == spk]
        by_speaker[spk] = baseline(group) if len(group) >= 4 else overall   # too few lines for a personal baseline

    refs_dir = episode_dir / "line_refs"
    refs_dir.mkdir(exist_ok=True)
    for seg in segments:
        loudness, f0 = feats[seg["id"]]
        seg["loudness_offset_db"] = round(loudness - by_speaker[seg.get("speaker")][0], 1)  # vs this speaker's usual
        if not seg.get("emotion") and seg["end"] - seg["start"] < MIN_JUDGE_SECONDS:
            seg["emotion"] = "neutral"
        if not seg.get("emotion"):
            s = score(loudness, f0, *by_speaker[seg.get("speaker")])
            seg["emotion"] = "intense" if s >= INTENSE_SCORE else "soft" if s <= SOFT_SCORE else "neutral"
            seg["emotion_score"] = round(s, 2)
        duration = seg["end"] - seg["start"]
        ref_path = refs_dir / f"{seg['id']:04d}.wav"
        if duration >= MIN_REF_SECONDS:
            if not ref_path.exists():
                start = max(0, round(seg["start"] * sr))
                sf.write(ref_path, mono[start: start + round(min(duration, MAX_REF_SECONDS) * sr)], sr)
            seg["line_reference"] = references.rel_to_root(ref_path)

    segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    return segments


if __name__ == "__main__":
    result = analyze(Path(sys.argv[1]))
    for seg in result:
        print(f"  id={seg['id']:>3} {seg.get('speaker')} {seg['emotion']:<8} score={seg.get('emotion_score')}  {seg['source_text'][:30]!r}")
