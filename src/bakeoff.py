"""M4 bake-off: dub the test clip with a voice-cloning engine, for judging by ear.

Usage: python src/bakeoff.py <engine_name> [source_episode_dir] [target_language]
  e.g. python src/bakeoff.py chatterbox work/vincenzo/test_clip en

Copies the source episode (already transcribed/diarized/translated) into
<show>/bakeoff_<engine>/, builds a reference clip per speaker from its
speaker_samples (separated vocals), synthesizes every line with the chosen
engine cloning that speaker, then runs the normal timing fit and mix, so the
result is directly comparable to the Kokoro dub (work/.../dubbed.mp4).
Per-line synth time and clip length are printed to help compare speed/fit.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import pathfix  # noqa: F401

REFERENCE_MAX_SECONDS = 15.0
COPY_FILES = ("source.mp4", "audio.wav", "vocals.wav", "no_vocals.wav", "speakers.json", "segments.json")


def build_references(src_dir: Path, dst_dir: Path, speakers: dict) -> dict[str, str]:
    """One reference wav per speaker: their sample clips joined, capped at ~15s."""
    ref_dir = dst_dir / "references"
    ref_dir.mkdir(exist_ok=True)
    refs = {}
    for spk, info in speakers.items():
        parts, total, sr = [], 0.0, None
        for clip in info["sample_clips"]:
            audio, sr = sf.read(src_dir / clip, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            parts.append(audio)
            parts.append(np.zeros(int(0.15 * sr), dtype="float32"))
            total += len(audio) / sr
            if total >= REFERENCE_MAX_SECONDS:
                break
        joined = np.concatenate(parts)[: int(REFERENCE_MAX_SECONDS * sr)]
        out = ref_dir / f"{spk}.wav"
        sf.write(out, joined, sr)
        refs[spk] = str(out)
        print(f"  reference {spk}: {len(joined) / sr:.1f}s")
    return refs


def main() -> None:
    engine_name = sys.argv[1]
    src_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "work/vincenzo/test_clip").resolve()
    target_language = sys.argv[3] if len(sys.argv) > 3 else "en"
    os.environ["DUBBER_TTS_ENGINE"] = engine_name

    dst_dir = src_dir.parent / f"bakeoff_{engine_name}_{target_language}"
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)
    for name in COPY_FILES:
        shutil.copyfile(src_dir / name, dst_dir / name)
    shutil.copytree(src_dir / "speaker_samples", dst_dir / "speaker_samples")

    speakers = json.loads((dst_dir / "speakers.json").read_text(encoding="utf-8"))
    refs = build_references(src_dir, dst_dir, speakers)

    segments = json.loads((dst_dir / "segments.json").read_text(encoding="utf-8"))
    for seg in segments:
        for key in ("audio_path", "voice", "timing_fit"):
            seg.pop(key, None)
        seg["reference"] = refs.get(seg.get("speaker"))
    (dst_dir / "segments.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    import mix
    import timing
    import tts
    import tts_engine

    t0 = time.time()
    tts.synthesize(dst_dir, None, target_language)
    print(f"synthesis: {time.time() - t0:.0f}s")
    timing.fit_timing(dst_dir, "ko", target_language)
    tts_engine.release_engines()
    mix.mix(dst_dir, "ko", target_language)
    print(f"done -> {dst_dir / 'dubbed.mp4'}")


if __name__ == "__main__":
    main()
