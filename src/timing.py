"""Stage: fit each TTS clip to its time slot before mixing.

Usage: python src/timing.py work/vincenzo/test_clip
Reads:  <episode_dir>/segments.json, <episode_dir>/tts/<id>.wav
Writes: updates segments.json ("english" may change if re-translated),
        rewrites tts/<id>.wav in place when sped up or re-synthesized shorter.

Tries, in order: fit as-is (the available slot already includes any natural
gap before the next line, i.e. "borrowed silence"), speed up the clip
(capped at config timing.max_speedup), ask the local LLM for a shorter
translation and re-synthesize once, then -- if it still doesn't fit --
flag needs_review and leave the overlap in place rather than fail the run.

A segment is never shortened against the next one if the original Korean
lines themselves overlapped (overlapping speech in the source), since the
dub is allowed to overlap there too.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from kokoro import KPipeline

import pathfix  # noqa: F401
from translate import call_ollama

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
WORDS_PER_SECOND = 2.3  # same rough estimate used in translate.py


def clip_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def speed_up(path: Path, factor: float) -> None:
    tmp = path.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-filter:a", f"atempo={factor:.4f}", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(path)


def ask_shorter_translation(korean: str, previous_english: str, target_words: int, model: str, host: str) -> str | None:
    prompt = (
        "You are translating Korean TV drama dialogue into English for dubbing. "
        "Your previous translation is too long to fit the scene's timing.\n\n"
        f'Korean (automatic transcription, may contain errors): "{korean}"\n'
        f'Previous English: "{previous_english}"\n\n'
        f"Give a SHORTER English translation of the same line, aiming for about {target_words} words "
        "or fewer, preserving meaning and tone as closely as possible.\n\n"
        'Respond with ONLY valid JSON: {"english": "<translation>"}'
    )
    try:
        raw = call_ollama(prompt, model, host)
        english = json.loads(raw)["english"].strip()
        return english or None
    except Exception:
        return None


def fit_segment(
    episode_dir: Path, segments: list[dict], i: int, max_speedup: float,
    ollama_model: str, ollama_host: str, tts_pipeline: KPipeline,
) -> None:
    """Mutates the clip on disk (speed-up and/or re-synthesis) at most once.
    Marks the segment "timing_fit": true on the way out in every case, so a
    resumed run (fit_timing, below) never re-applies speed-up to an already
    sped-up clip -- atempo is not idempotent, re-running it would compound."""
    seg = segments[i]
    if not seg.get("audio_path"):
        return

    try:
        is_last = i + 1 >= len(segments)
        original_overlap = (not is_last) and segments[i + 1]["start"] < seg["end"]
        if is_last or original_overlap:
            return  # nothing after it to collide with, or original speech overlapped here too

        available = segments[i + 1]["start"] - seg["start"]
        clip_path = episode_dir / seg["audio_path"]
        duration = clip_duration(clip_path)
        if duration <= available:
            return

        needed_factor = duration / available
        if needed_factor <= max_speedup:
            speed_up(clip_path, needed_factor)
            print(f"  id={seg['id']}: sped up {needed_factor:.2f}x to fit {available:.2f}s slot")
            return

        target_words = max(1, round(available * WORDS_PER_SECOND))
        shorter = ask_shorter_translation(seg["korean"], seg["english"], target_words, ollama_model, ollama_host)
        if shorter:
            chunks = [audio for _, _, audio in tts_pipeline(shorter, voice=seg["voice"])]
            audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            sf.write(clip_path, audio, 24000)
            seg["english"] = shorter
            # A compressed re-translation isn't always natural English (caught
            # one that read as broken, e.g. "Thirds out, dangerous.") -- flag
            # it so it gets a human look even if it technically fits the slot.
            seg["needs_review"] = True

            new_duration = clip_duration(clip_path)
            if new_duration <= available:
                print(f"  id={seg['id']}: shorter translation fit {available:.2f}s slot without speedup -- flagged for review")
                return
            factor = new_duration / available
            if factor <= max_speedup:
                speed_up(clip_path, factor)
                print(f"  id={seg['id']}: shorter translation + sped up {factor:.2f}x to fit {available:.2f}s slot -- flagged for review")
                return

        seg["needs_review"] = True
        print(f"  id={seg['id']}: COULD NOT FIT ({duration:.2f}s into {available:.2f}s slot) -- flagged needs_review")
    finally:
        seg["timing_fit"] = True


def fit_timing(episode_dir: Path) -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text())
    max_speedup = config["timing"]["max_speedup"]
    ollama_cfg = config["models"]["ollama"]

    tts_pipeline = KPipeline(lang_code="a")

    todo = [i for i, s in enumerate(segments) if not s.get("timing_fit")]
    if len(todo) < len(segments):
        print(f"  resuming: {len(segments) - len(todo)}/{len(segments)} segments already fitted, skipping those")

    for n, i in enumerate(todo):
        fit_segment(episode_dir, segments, i, max_speedup, ollama_cfg["model"], ollama_cfg["host"], tts_pipeline)
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            print(f"  timing fit: {n + 1}/{len(todo)}")
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    fit_timing(episode_dir)
    print("timing fit done")
