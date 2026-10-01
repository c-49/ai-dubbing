# Kdrama English Dubbing Pipeline: Project Plan

## Goal
Take a Korean drama episode (mp4, 30-60 min, no subtitles) and produce an mp4 with English dubbed voices. Personal, offline use only (for the user and their mother). Never distributed or sold.

## Hard constraints
- **Everything must be free.** No paid APIs, no paid services. Do not add a dependency that requires payment or a credit card.
- **Windows 11, 16GB RAM, AMD GPU (no NVIDIA/CUDA).** Target CPU. Long runtimes (hours per episode) are acceptable.
- **User is comfortable with Python but not an expert.** Explain setup steps, keep code readable, avoid clever abstractions.
- Work one milestone at a time. Stop after each milestone, show what was built, and let the user test before moving on.

## How to work with the user
- Start with Milestone 0 (setup). Do not write pipeline code until setup is verified.
- Test on a 5-minute clip first, never a full episode. The user will provide the clip.
- Before installing anything large (models, torch), say what it is and roughly how big it is.
- If a tool doesn't work on Windows/CPU, say so and propose an alternative rather than silently working around it.

## Pipeline overview
Each stage reads and writes files in a per-episode work folder, so any stage can be re-run, and the user can hand-edit between stages.

1. **Extract audio** (ffmpeg) -> `audio.wav`
2. **Separate** (Demucs) -> `vocals.wav` and `no_vocals.wav` (music/effects)
3. **Transcribe Korean** (faster-whisper large-v3, CPU, int8, language=ko, VAD on) using `vocals.wav` -> timed Korean segments
4. **Speaker diarization** (pyannote.audio) -> speaker label per segment, plus a guess of male/female per speaker
5. **Translate** (local LLM via Ollama, 7-9B model such as Qwen2.5 7B or Gemma 2) -> English per segment, using a glossary and context chunks
6. **Review** (local web page) -> user fixes lines, assigns voices
7. **TTS** (Kokoro, fallback Piper) -> one audio clip per segment
8. **Timing fit** -> adjust each clip to its time slot
9. **Mix** (ffmpeg) -> original music/effects at normal volume, faint original vocals, English dub on top
10. **Export** -> mp4 with the video stream copied, not re-encoded

## Central data file: `segments.json`
One record per spoken line. Every stage adds or updates fields.

```json
{
  "id": 12,
  "start": 83.42,
  "end": 86.10,
  "speaker": "SPEAKER_01",
  "korean": "...",
  "english": "...",
  "voice": "af_bella",
  "audio_path": "tts/0012.wav",
  "needs_review": false
}
```

Also per episode: `speakers.json` (speaker -> voice choice, gender guess, sample clip times). Also per show: `glossary.json` (character names, recurring terms, honorific preferences) and `voices.json` (saved speaker-to-voice mapping, reused across episodes).

## Suggested project layout
```
dubber/
  PLAN.md
  README.md
  requirements.txt
  config.yaml            # paths, model sizes, mix levels
  src/
    extract.py
    separate.py
    transcribe.py
    diarize.py
    translate.py
    tts.py
    timing.py
    mix.py
    run_episode.py       # runs stages in order, skipping finished ones
  review_ui/             # local web page (later milestone)
  shows/
    <show-name>/
      glossary.json
      voices.json
  work/
    <episode-name>/      # all intermediate files
```

## Milestones

### Milestone 0: Setup
- Check/install Python 3.10-3.12, ffmpeg (on PATH), Git. Create a virtual environment.
- Install PyTorch **CPU build**, faster-whisper, demucs, pyannote.audio, and the TTS package.
- Install Ollama and pull one 7-9B model.
- pyannote needs a free Hugging Face account, an access token, and accepting the model terms once. Walk the user through this.
- Kokoro needs espeak-ng on Windows. Verify this works.
- **Done when:** a tiny test script runs each tool once (ffmpeg version, a Whisper transcription of a 10-second sample, an Ollama reply, a Kokoro audio file).

### Milestone 1: Transcription on the test clip
- Extract audio, run Demucs, transcribe `vocals.wav` with faster-whisper.
- Output `segments.json` with start, end, and Korean text.
- Pass the glossary (if any) as an initial prompt hint.
- **Done when:** the user reads the Korean output and judges accuracy. Record notes on hallucinated lines over music/silence and tune VAD settings.

### Milestone 2: Translation
- Translate in chunks of about 20-30 segments with a few lines of surrounding context, speaker labels if available, and the glossary.
- Prompt must say the Korean is automatic transcription that may contain errors, and to infer from context. Preserve tone and register, keep names consistent.
- Ask for a **target length** per line (roughly matching the Korean line's duration) to ease timing later. Output must be strict JSON that maps back to segment ids. Validate it and retry on malformed output.
- **Done when:** every segment has English and the user approves quality on the test clip.

### Milestone 3: One-voice end-to-end
- Generate TTS for each segment with a single voice, place clips at their start times, mix with `no_vocals.wav`, export mp4.
- Keep the original audio as a second audio track so the user can switch.
- **Done when:** the user can play a 5-minute dubbed clip. Quality can be rough, but the whole path must work.

### Milestone 4: Speakers and voices
- Run pyannote diarization, assign speakers to segments by overlap.
- Guess gender per speaker from pitch (e.g. median F0) as a suggestion only; the user can override.
- For each speaker, export 2-3 sample clips so the user can hear who they are.
- Let the user pick an English voice per speaker (list available Kokoro voices with a labeled male/female). Save to `voices.json` for reuse on later episodes of the same show.
- Handle overlapping speech and short low-confidence segments by flagging `needs_review`.

### Milestone 5: Timing and mixing
- Measure each TTS clip against its time slot. In order of preference: use the shorter translation, speed up TTS slightly (cap at about 1.25x), borrow from adjacent silence, then flag the line for a shorter re-translation.
- Never let two dub clips overlap unless the original speakers overlap.
- Ducking: lower the original vocal track while English plays. Suggested starting levels: music/effects 100%, original vocals about 10-15%, English dub about 100%. Put levels in `config.yaml`.
- Normalize final loudness.
- **Done when:** a dubbed clip sounds natural and dialogue is easy to hear over the music.

### Milestone 6: Review page
- Small local web page (Flask or FastAPI plus plain HTML/JS) listing segments: Korean, English (editable), speaker, voice dropdown, play button for the TTS clip, `needs_review` filter.
- Editing a line regenerates only that clip. Include a "re-run mix" button.
- Show speaker sample clips and voice selection here.

### Milestone 7: Full-episode run
- Run on a complete 1-hour episode overnight. Add progress logging, resume-after-crash (skip finished stages and segments), and a rough time estimate per stage.
- Document the full workflow in `README.md` in plain steps.

## Known risks (keep in mind throughout)
- **Whisper hallucination** over music or silence: mitigate with Demucs vocals-only input, VAD, and dropping segments with very low confidence.
- **Stacked errors:** wrong Korean -> wrong English. The review page and glossary are the defense.
- **Diarization mistakes** on overlapping speech, crying, shouting.
- **Flat TTS emotion.** Accept this; do not chase it until the rest works.
- **Local LLM translation quality** is lower than top paid models. Mitigate with small chunks, context, glossary, and review. (The user may also paste hard lines into Claude chat manually, which is allowed and free, but is not part of the automated pipeline.)
- **RAM:** 16GB total. Run stages sequentially, free models between stages, and do not load Whisper and the LLM at the same time.

## Out of scope (for now)
- GPU acceleration (revisit when the user gets new hardware; whisper.cpp with Vulkan is the likely AMD route).
- Voice cloning, lip-sync, emotion-controlled TTS.
- Any hosting, sharing, or distribution.
