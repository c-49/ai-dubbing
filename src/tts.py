"""Stage: generate TTS audio for each segment, one voice per speaker.

Usage: python src/tts.py work/vincenzo/test_clip [path/to/voices.json]
Reads:  <episode_dir>/segments.json  (needs "english" filled in)
Writes: <episode_dir>/tts/<id>.wav, updates segments.json with "voice"/"audio_path"

If no voices.json is given, or a segment's speaker isn't in it, falls back
to config.yaml's models.tts.default_voice.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from kokoro import KPipeline

import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
SAMPLE_RATE = 24000


def synthesize(episode_dir: Path, voices_path: Path | None = None) -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    default_voice = yaml.safe_load(CONFIG_PATH.read_text())["models"]["tts"]["default_voice"]
    speaker_voices = {}
    if voices_path and voices_path.exists():
        speaker_voices = json.loads(voices_path.read_text(encoding="utf-8"))

    tts_dir = episode_dir / "tts"
    tts_dir.mkdir(exist_ok=True)

    pipeline = KPipeline(lang_code="a")  # American English

    todo = [
        seg for seg in segments
        if seg.get("english") and not (seg.get("audio_path") and (episode_dir / seg["audio_path"]).exists())
    ]
    done_count = sum(1 for s in segments if s.get("english")) - len(todo)
    if done_count:
        print(f"  resuming: {done_count} clips already generated, skipping those")

    for n, seg in enumerate(todo):
        voice = speaker_voices.get(seg.get("speaker"), default_voice)
        chunks = [audio for _, _, audio in pipeline(seg["english"], voice=voice)]
        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        out_path = tts_dir / f"{seg['id']:04d}.wav"
        sf.write(out_path, audio, SAMPLE_RATE)
        seg["voice"] = voice
        seg["audio_path"] = str(out_path.relative_to(episode_dir)).replace("\\", "/")
        print(f"  id={seg['id']} ({seg.get('speaker')}, {voice}): {len(audio) / SAMPLE_RATE:.2f}s -- {seg['english']!r}")
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            print(f"  TTS: {n + 1}/{len(todo)}")
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    voices_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = synthesize(episode_dir, voices_path)
    print(f"generated TTS for {sum(1 for s in result if s.get('audio_path'))} segments")
