"""Stage 3: transcribe Korean speech with faster-whisper.

Usage: python src/transcribe.py work/vincenzo/test_clip
Reads:  <episode_dir>/vocals.wav
Writes: <episode_dir>/segments.json

This is likely the single longest stage on a full episode (hours on CPU),
so it supports resuming after a crash: progress is saved every 20 segments,
and if interrupted, the next run re-transcribes only from the last saved
segment's end time onward (by trimming a temp copy of vocals.wav) rather
than starting over. A ".transcribe_done" marker distinguishes "finished" from
"partial" -- segments.json existing alone doesn't mean the stage is done.
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import yaml
from faster_whisper import WhisperModel

import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def transcribe(episode_dir: Path, glossary_terms: list[str] | None = None) -> list[dict]:
    vocals_in = episode_dir / "vocals.wav"
    if not vocals_in.exists():
        raise FileNotFoundError(f"no vocals at {vocals_in}, run separate.py first")

    segments_path = episode_dir / "segments.json"
    done_marker = episode_dir / ".transcribe_done"
    if done_marker.exists():
        return json.loads(segments_path.read_text(encoding="utf-8"))

    records = json.loads(segments_path.read_text(encoding="utf-8")) if segments_path.exists() else []
    resume_from = max((r["end"] for r in records), default=0.0)
    next_id = (records[-1]["id"] + 1) if records else 0

    trimmed_path = None
    if resume_from > 0:
        print(f"  resuming transcription from {resume_from:.1f}s ({len(records)} segments already done)")
        audio, sr = sf.read(vocals_in, dtype="float32", always_2d=True)
        trimmed_path = episode_dir / "_resume_vocals.wav"
        sf.write(trimmed_path, audio[round(resume_from * sr):], sr)

    config = yaml.safe_load(CONFIG_PATH.read_text())["models"]["whisper"]
    model = WhisperModel(config["size"], device=config["device"], compute_type=config["compute_type"])

    initial_prompt = ", ".join(glossary_terms) if glossary_terms else None
    segments, info = model.transcribe(
        str(trimmed_path or vocals_in),
        language="ko",
        vad_filter=True,
        vad_parameters=config["vad"],
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
    )

    # Flag likely hallucinations: a long stretch transcribed as very little
    # text is the signature of Whisper filling silence/noise with filler
    # syllables. (no_speech_prob/avg_logprob turned out too coarse for this --
    # under vad_filter, faster-whisper decodes all speech concatenated into
    # ~30s windows, so those metrics are shared across many segments at once
    # and don't isolate the bad fragment within a window.)
    MIN_FLAG_DURATION_S = 2.5
    MIN_CHARS_PER_SECOND = 0.6

    for seg in segments:
        start = seg.start + resume_from
        end = seg.end + resume_from
        text = seg.text.strip()
        duration = end - start
        char_count = len(text.replace(" ", "").rstrip(".?!"))
        chars_per_second = char_count / duration if duration > 0 else 0
        flagged = duration >= MIN_FLAG_DURATION_S and chars_per_second < MIN_CHARS_PER_SECOND
        if flagged:
            print(f"  flagged id={next_id} ({start:.1f}-{end:.1f}s, {chars_per_second:.2f} chars/sec): {text!r}")
        records.append({
            "id": next_id,
            "start": round(start, 2),
            "end": round(end, 2),
            "speaker": None,
            "korean": text,
            "english": None,
            "voice": None,
            "audio_path": None,
            "needs_review": flagged,
        })
        next_id += 1
        if next_id % 20 == 0:
            print(f"  transcribed up to {end:.0f}s, {next_id} segments so far")
            segments_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    segments_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if trimmed_path:
        trimmed_path.unlink()
    done_marker.write_text("")
    return records


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    records = transcribe(episode_dir)
    print(f"wrote {len(records)} segments to {episode_dir / 'segments.json'}")
