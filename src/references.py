"""Reference clips for voice cloning: one short sample of each speaker's
original voice, cut from the separated vocals.

An episode gets an automatically built reference per speaker
(work/<show>/<episode>/references/<speaker>.wav). A reviewer can override it
for the whole show with a clip they picked in the review UI, saved as
shows/<show>/references/<speaker>.wav; that one wins in every episode.

Paths stored in segments.json are project-root-relative with forward
slashes, so they survive moving the project folder (tts_engine resolves them).
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT_DIR = Path(__file__).resolve().parent.parent
SHOWS_DIR = ROOT_DIR / "shows"

TARGET_SECONDS = 15.0     # total reference length to aim for
MIN_CLIP_SECONDS = 2.0    # ignore fragments shorter than this
MAX_CLIP_SECONDS = 9.0    # a long monologue clip is trimmed to this
GAP_SECONDS = 0.2


def rel_to_root(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT_DIR).as_posix()


def build_reference(episode_dir: Path, speaker: str, segments: list[dict]) -> Path | None:
    """Joins this speaker's cleanest lines (not flagged, not ignored, in the
    2-9s range, preferring ones nearest 5s) into ~15s of audio."""
    vocals_path = episode_dir / "vocals.wav"
    if not vocals_path.exists():
        return None
    candidates = [
        s for s in segments
        if s.get("speaker") == speaker and not s.get("needs_review") and not s.get("ignored")
        and s["end"] - s["start"] >= MIN_CLIP_SECONDS
    ]
    if not candidates:  # speaker has no clean lines: fall back to any of their lines
        candidates = [s for s in segments if s.get("speaker") == speaker
                      and s["end"] - s["start"] >= MIN_CLIP_SECONDS]
    if not candidates:
        return None
    candidates.sort(key=lambda s: abs((s["end"] - s["start"]) - 5.0))

    audio, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    parts, total = [], 0.0
    for seg in candidates:
        end = min(seg["end"], seg["start"] + MAX_CLIP_SECONDS)
        clip = mono[round(seg["start"] * sr):round(end * sr)]
        parts += [clip, np.zeros(round(GAP_SECONDS * sr), dtype="float32")]
        total += len(clip) / sr + GAP_SECONDS
        if total >= TARGET_SECONDS:
            break
    out_dir = episode_dir / "references"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{speaker}.wav"
    sf.write(out_path, np.concatenate(parts)[: round(TARGET_SECONDS * sr)], sr)
    return out_path


def ensure_references(episode_dir: Path) -> dict[str, str]:
    """Builds any missing automatic reference. Returns {speaker: root-relative path}."""
    segments = json.loads((episode_dir / "segments.json").read_text(encoding="utf-8"))
    speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})
    refs = {}
    for speaker in speakers:
        path = episode_dir / "references" / f"{speaker}.wav"
        if not path.exists():
            path = build_reference(episode_dir, speaker, segments)
        if path:
            refs[speaker] = rel_to_root(path)
    return refs


def resolve_reference(show: str, episode_dir: Path, speaker: str) -> str | None:
    """Show-level override first, then the episode's automatic reference."""
    override = SHOWS_DIR / show / "references" / f"{speaker}.wav"
    if override.exists():
        return rel_to_root(override)
    auto = episode_dir / "references" / f"{speaker}.wav"
    return rel_to_root(auto) if auto.exists() else None
