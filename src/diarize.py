"""Stage 4: speaker diarization with pyannote, assigned to segments by overlap.

Usage: python src/diarize.py work/vincenzo/test_clip
Reads:  <episode_dir>/vocals.wav, <episode_dir>/segments.json
Writes: updates segments.json with "speaker", writes speakers.json
"""
import json
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as taf
import yaml
from pyannote.audio import Pipeline

import hardware
import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# Rough median-F0 cutover between typical male/female speaking voices, used
# only as a starting suggestion -- the user picks the actual voice either way.
GENDER_F0_CUTOVER_HZ = 165.0
MIN_VOICED_HZ, MAX_VOICED_HZ = 60.0, 400.0
SAMPLE_CLIPS_PER_SPEAKER = 3


def load_token() -> str:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if not token_file.exists():
        raise RuntimeError("no Hugging Face token found -- see README.md setup steps")
    return token_file.read_text().strip()


def overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_same_speaker_turns(turns: list[dict], max_gap: float = 1.0) -> list[dict]:
    """pyannote fragments continuous same-speaker speech into many short turns
    separated by tiny pauses. Treating each fragment as independent makes
    "how many distinct speakers overlap this segment" noisy and sensitive to
    run-to-run jitter in turn boundaries. Merging same-speaker turns that are
    close together gives a stable signal for genuine cross-speaker overlap."""
    turns = sorted(turns, key=lambda t: t["start"])
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] - merged[-1]["end"] <= max_gap:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    return merged


def estimate_gender(mono_audio: torch.Tensor, sample_rate: int) -> tuple[str | None, float | None]:
    if mono_audio.numel() < sample_rate // 2:  # less than ~0.5s, too little to trust
        return None, None
    pitch = taf.detect_pitch_frequency(mono_audio, sample_rate)
    voiced = pitch[(pitch > MIN_VOICED_HZ) & (pitch < MAX_VOICED_HZ)]
    if voiced.numel() == 0:
        return None, None
    median_f0 = float(voiced.median())
    gender = "male" if median_f0 < GENDER_F0_CUTOVER_HZ else "female"
    return gender, round(median_f0, 1)


def export_speaker_samples(
    episode_dir: Path, speaker: str, spk_segments: list[dict], mono_audio: torch.Tensor, sample_rate: int
) -> list[str]:
    samples_dir = episode_dir / "speaker_samples"
    samples_dir.mkdir(exist_ok=True)

    # Longest segments make the clearest samples -- but skip ones already
    # flagged needs_review (e.g. hallucinated transcription over noise),
    # which wouldn't be representative of the speaker's actual voice.
    clean_segments = [s for s in spk_segments if not s.get("needs_review")] or spk_segments
    picks = sorted(clean_segments, key=lambda s: s["end"] - s["start"], reverse=True)[:SAMPLE_CLIPS_PER_SPEAKER]
    paths = []
    for i, seg in enumerate(picks):
        start_sample = round(seg["start"] * sample_rate)
        end_sample = round(seg["end"] * sample_rate)
        clip = mono_audio[start_sample:end_sample].numpy()
        out_path = samples_dir / f"{speaker}_{i + 1:02d}.wav"
        sf.write(out_path, clip, sample_rate)
        paths.append(str(out_path.relative_to(episode_dir)).replace("\\", "/"))
    return paths


def diarize(episode_dir: Path) -> tuple[list[dict], dict]:
    vocals_path = episode_dir / "vocals.wav"
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    speakers_path = episode_dir / "speakers.json"
    if speakers_path.exists():
        return segments, json.loads(speakers_path.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text())["models"]["pyannote"]
    pipeline = Pipeline.from_pretrained(config["pipeline"], token=load_token())
    device = hardware.resolve_device(config.get("device", "auto"))
    print(f"  pyannote: {device}")
    pipeline.to(torch.device(device))

    # Load the audio ourselves and hand pyannote a waveform tensor directly,
    # bypassing its torchcodec-based file decoder (buggy on this Windows/CPU
    # setup -- see pathfix.py comments for the related DLL-loading fix).
    audio_np, sample_rate = sf.read(vocals_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_np.T)  # (channel, time)
    result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in result.speaker_diarization.itertracks(yield_label=True)
    ]
    del pipeline, result
    hardware.release_gpu()
    turns = merge_same_speaker_turns(turns)

    for seg in segments:
        overlaps = [
            (overlap(seg["start"], seg["end"], t["start"], t["end"]), t["speaker"])
            for t in turns
        ]
        overlaps = [(ov, spk) for ov, spk in overlaps if ov > 0]

        if not overlaps:
            seg["speaker"] = None
            seg["needs_review"] = True
            continue

        best_overlap, best_speaker = max(overlaps, key=lambda x: x[0])
        seg["speaker"] = best_speaker

        # Whisper and pyannote use different boundary granularities, so a
        # low overlap ratio alone is normal and not a useful signal (checked
        # empirically -- it flagged nearly every segment). Multiple distinct
        # speakers overlapping the same segment is the real ambiguity signal.
        distinct_speakers = {spk for _, spk in overlaps}
        if len(distinct_speakers) > 1:
            seg["needs_review"] = True

    segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    mono_audio = waveform.mean(dim=0)  # (time,), for pitch estimation and sample clips

    speakers = sorted({s["speaker"] for s in segments if s["speaker"]})
    speaker_stats = {}
    for spk in speakers:
        spk_segments = [s for s in segments if s["speaker"] == spk]

        voiced_slices = [
            mono_audio[round(s["start"] * sample_rate):round(s["end"] * sample_rate)]
            for s in spk_segments
        ]
        gender_guess, median_f0 = estimate_gender(torch.cat(voiced_slices), sample_rate)

        speaker_stats[spk] = {
            "segment_count": len(spk_segments),
            "total_duration": round(sum(s["end"] - s["start"] for s in spk_segments), 2),
            "gender_guess": gender_guess,
            "median_f0_hz": median_f0,
            "sample_clips": export_speaker_samples(episode_dir, spk, spk_segments, mono_audio, sample_rate),
            "voice": None,
        }

    speakers_path = episode_dir / "speakers.json"
    speakers_path.write_text(json.dumps(speaker_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return segments, speaker_stats


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    segments, speakers = diarize(episode_dir)
    print(f"found {len(speakers)} speakers: {list(speakers.keys())}")
    for spk, stats in speakers.items():
        print(
            f"  {spk}: {stats['segment_count']} segments, {stats['total_duration']:.1f}s total, "
            f"gender_guess={stats['gender_guess']} (median F0={stats['median_f0_hz']} Hz), "
            f"samples={stats['sample_clips']}"
        )
