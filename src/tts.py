"""Stage: generate TTS audio for each segment, one voice per speaker.

Usage: python src/tts.py work/vincenzo/test_clip [path/to/voices.json] [target_language]
Reads:  <episode_dir>/segments.json  (needs "target_text" filled in)
Writes: <episode_dir>/tts/<id>.wav, updates segments.json with "voice"/"audio_path"

If no voices.json is given, or a segment's speaker isn't in it, falls back
to config.yaml's languages.target.<target_language>.default_voice.
target_language defaults to "en" when omitted, e.g. for ad-hoc CLI runs.
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import yaml

import pathfix  # noqa: F401
import tts_engine

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def synthesize(episode_dir: Path, voices_path: Path | None = None, target_language: str = "en") -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    target_cfg = yaml.safe_load(CONFIG_PATH.read_text())["languages"]["target"][target_language]
    default_voice = target_cfg["default_voice"]
    speaker_voices = {}
    if voices_path and voices_path.exists():
        speaker_voices = json.loads(voices_path.read_text(encoding="utf-8"))

    tts_dir = episode_dir / "tts"
    tts_dir.mkdir(exist_ok=True)

    engine = tts_engine.get_engine(target_language)

    # "ignored" segments keep their source/target text and timing (useful if
    # a line looked like a transcription hallucination but might not be --
    # it's recoverable) but are deliberately left un-dubbed: no clip, so
    # mix.py never places anything for them.
    dubbable = [s for s in segments if s.get("target_text") and not s.get("ignored")]
    todo = [seg for seg in dubbable if not (seg.get("audio_path") and (episode_dir / seg["audio_path"]).exists())]
    done_count = len(dubbable) - len(todo)
    if done_count:
        print(f"  resuming: {done_count} clips already generated, skipping those")

    for n, seg in enumerate(todo):
        voice = speaker_voices.get(seg.get("speaker"), default_voice)
        audio = engine.synthesize(seg["target_text"], voice, emotion=seg.get("emotion"), reference=seg.get("reference"))
        out_path = tts_dir / f"{seg['id']:04d}.wav"
        sf.write(out_path, audio, engine.sample_rate)
        seg["voice"] = voice
        seg["audio_path"] = str(out_path.relative_to(episode_dir)).replace("\\", "/")
        print(f"  id={seg['id']} ({seg.get('speaker')}, {voice}): {len(audio) / engine.sample_rate:.2f}s -- {seg['target_text']!r}")
        if (n + 1) % 20 == 0 or n + 1 == len(todo):
            print(f"  TTS: {n + 1}/{len(todo)}")
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    voices_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    target_language = sys.argv[3] if len(sys.argv) > 3 else "en"
    result = synthesize(episode_dir, voices_path, target_language)
    print(f"generated TTS for {sum(1 for s in result if s.get('audio_path'))} segments")
