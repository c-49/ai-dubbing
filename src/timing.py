"""Stage: fit each TTS clip to its time slot before mixing.

Usage: python src/timing.py work/vincenzo/test_clip
Reads:  <episode_dir>/segments.json, <episode_dir>/tts/<id>.wav
Writes: updates segments.json ("target_text" may change if re-translated),
        rewrites tts/<id>.wav in place when sped up or re-synthesized shorter.

Tries, in order: fit as-is (the available slot already includes any natural
gap before the next line, i.e. "borrowed silence"), speed up the clip
(capped at config timing.max_speedup), ask the local LLM for a shorter
translation and re-synthesize once, then -- if it still doesn't fit --
flag needs_review and leave the overlap in place rather than fail the run.

A segment is never shortened against the next one if the original source
lines themselves overlapped (overlapping speech in the source), since the
dub is allowed to overlap there too.
"""
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import yaml
import tts_engine

import hardware
import pathfix  # noqa: F401
from translate import call_ollama

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


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


def ask_shorter_translation(
    source_text: str, previous_translation: str, target_words: int,
    source_name: str, target_name: str, model: str, host: str,
) -> str | None:
    prompt = (
        f"You are translating {source_name} TV drama dialogue into {target_name} for dubbing. "
        "Your previous translation is too long to fit the scene's timing.\n\n"
        f'{source_name} (automatic transcription, may contain errors): "{source_text}"\n'
        f'Previous {target_name}: "{previous_translation}"\n\n'
        f"Give a SHORTER {target_name} translation of the same line, aiming for about {target_words} words "
        "or fewer, preserving meaning and tone as closely as possible.\n\n"
        'Respond with ONLY valid JSON: {"translation": "<translation>"}'
    )
    try:
        raw = call_ollama(prompt, model, host)
        translation = json.loads(raw)["translation"].strip()
        return translation or None
    except Exception:
        return None


def slot_seconds(segments: list[dict], i: int) -> float | None:
    """Seconds available to segment i before the next line starts, or None if
    it is last or the original speech overlapped the next line too."""
    if i + 1 >= len(segments) or segments[i + 1]["start"] < segments[i]["end"]:
        return None
    return segments[i + 1]["start"] - segments[i]["start"]


def fit_segment(
    episode_dir: Path, segments: list[dict], i: int, max_speedup: float,
    ollama_model: str, ollama_host: str, target_language: str,
    source_name: str = "Korean", target_name: str = "English", words_per_second: float = 2.3,
    allow_auto_shorten: bool = True,
) -> None:
    """Mutates the clip on disk (speed-up and/or re-synthesis) at most once.
    Marks the segment "timing_fit": true on the way out in every case, so a
    resumed run (fit_timing, below) never re-applies speed-up to an already
    sped-up clip -- atempo is not idempotent, re-running it would compound.

    allow_auto_shorten gates the last-resort "ask the LLM for a shorter
    translation" step. True during the unsupervised pipeline run
    (fit_timing) where nobody has looked at the line yet. The review UI
    passes False when re-fitting right after a human hand-edited this
    exact line's text -- auto-shortening then would silently overwrite
    their edit with another automated rewrite, defeating the point of
    editing it by hand."""
    seg = segments[i]
    if not seg.get("audio_path") or seg.get("ignored"):
        return  # no clip to fit, or deliberately left un-dubbed (see tts.py)

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

        cloned = seg.get("voice") == tts_engine.CLONE_VOICE
        if cloned:
            # Ollama (~4GB) and a cloning model together overflow a 6GB card,
            # which silently spills to system RAM and runs several times slower.
            tts_engine.release_engines()
        shorter = ask_shorter_translation(
            seg["source_text"], seg["target_text"], max(1, round(available * words_per_second)),
            source_name, target_name, ollama_model, ollama_host,
        ) if allow_auto_shorten else None
        if cloned:
            hardware.unload_ollama(ollama_model, ollama_host)
        if shorter:
            audio, sample_rate = tts_engine.synthesize_segment(seg, shorter, target_language, duration=available)
            sf.write(clip_path, audio, sample_rate)
            seg["target_text"] = shorter
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


def fit_timing(episode_dir: Path, source_language: str = "ko", target_language: str = "en") -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text())
    max_speedup = config["timing"]["max_speedup"]
    ollama_cfg = config["models"]["ollama"]
    languages = config["languages"]
    source_name = languages["source"][source_language]["name"]
    target_cfg = languages["target"][target_language]
    target_name = target_cfg["name"]
    words_per_second = target_cfg["words_per_second"]


    todo = [i for i, s in enumerate(segments) if not s.get("timing_fit")]
    if len(todo) < len(segments):
        print(f"  resuming: {len(segments) - len(todo)}/{len(segments)} segments already fitted, skipping those")

    for n, i in enumerate(todo):
        fit_segment(
            episode_dir, segments, i, max_speedup, ollama_cfg["model"], ollama_cfg["host"], target_language,
            source_name, target_name, words_per_second,
        )
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            print(f"  timing fit: {n + 1}/{len(todo)}")
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    hardware.unload_ollama(ollama_cfg["model"], ollama_cfg["host"])
    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    source_language = sys.argv[2] if len(sys.argv) > 2 else "ko"
    target_language = sys.argv[3] if len(sys.argv) > 3 else "en"
    fit_timing(episode_dir, source_language, target_language)
    print("timing fit done")
