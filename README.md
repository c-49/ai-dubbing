# Kdrama English Dubbing Pipeline

See `PLAN.md` for the full project plan. This file only covers day-to-day setup
and running instructions, updated as milestones land.

## Setup (Milestone 0)

Already done by the assistant on this machine:
- Python 3.11 installed (alongside whatever other Python versions you have), via winget.
- ffmpeg installed via winget (`ffmpeg`, `ffprobe` on PATH).
- eSpeak NG installed via winget (Kokoro TTS needs it on Windows).
- Virtual environment created at `venv/` using Python 3.11.
- Installed into `venv/`: PyTorch (CPU build), faster-whisper, demucs, pyannote.audio, kokoro.
- Ollama (already installed) has two models pulled: `llama3:latest` and `qwen2.5:7b`
  (qwen2.5:7b is the default translation model in `config.yaml` -- better at Korean).

### One thing only you can do: Hugging Face token for pyannote

Speaker diarization (Milestone 4) uses a gated model on Hugging Face. To use it:

1. Create a free account at https://huggingface.co/join (skip if you have one).
2. Accept the model terms on these three pages (just visit and click "Agree"):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
     (the `pyannote.audio` version we installed, 4.0.7, pulls this in as a
     dependency of the 3.1 pipeline even though it's not in the pipeline name --
     an upstream change after this plan was written)
3. Create an access token: https://huggingface.co/settings/tokens -> "New token" ->
   type "Read" is enough.
4. Log in from this machine so the token is saved locally:
   ```
   venv\Scripts\huggingface-cli login
   ```
   Paste the token when prompted. This writes the token to
   `~/.cache/huggingface/token`, which `check_setup.py` and the diarization
   script look for.

You don't need this for Milestones 0-3. It's only required starting Milestone 4.

## Running the setup check

```
venv\Scripts\python.exe src\check_setup.py
```

This runs ffmpeg, generates a short Kokoro audio clip, transcribes it back with
faster-whisper (using the small "base" model, just to prove the mechanics work --
the real pipeline uses large-v3, downloaded the first time Milestone 1 runs),
asks Ollama for a one-word reply, and (once the HF token above is set up) loads
the pyannote diarization pipeline once.

## Review page (Milestone 6)

A local web app for browsing shows/episodes and reviewing/fixing a dubbed
episode before final export:

```
venv\Scripts\python.exe review_ui\app.py
```

Then open http://127.0.0.1:5000 in a browser. The home page lists shows
(`shows/<name>/`), each show page lists its episodes (`work/<show>/<episode>/`),
and clicking an episode opens its review page, which shows:
- **Speaker cards** at the top: gender guess, sample clips to listen to, and a
  voice dropdown -- changing it regenerates every line for that speaker and
  saves the choice to `shows/<show>/voices.json` for future episodes.
- **A table of every segment**: Korean text, editable English text, a
  per-line voice override, a Play button for that line's TTS clip, and a
  "needs review" checkbox you can set or clear by hand.
- **"Show only flagged"** checkbox to filter down to lines the pipeline
  flagged automatically (hallucination risk, timing conflicts, etc.)
- **"Re-run mix"** button at the top: regenerates `dubbed.mp4` from the
  current state of all segments, so you can check your edits in context.

Editing English text or changing a voice regenerates just that one clip
(and re-checks it still fits its time slot) -- it does not touch anything
else. Close the page and stop the server with Ctrl+C in its terminal when
you're done; your edits are already saved to `segments.json` as you go.

## Running a full episode (Milestone 7)

Episodes live at `work/<show-name>/<episode-name>/` -- the show name is just
the parent folder, so `work/vincenzo/ep5/` means show "vincenzo", episode "ep5".

### First time setting up a new show

1. Create `shows/<show-name>/incoming/` (just an empty folder -- this is
   where you'll drop new videos for this show).
2. Create `shows/<show-name>/glossary.json` (optional, but recommended) --
   a flat `{"Korean term": "English translation"}` mapping of character
   names and recurring terms. See `shows/vincenzo/glossary.json` for an
   example. Skip it for the first episode if you don't know the names yet;
   add it later once you do.
3. There's no need to create `voices.json` by hand -- the review page writes
   it automatically the first time you pick voices for that show's speakers.

### Per episode: drop it in, then run one command

1. Drop the video into that show's inbox:
   `shows/<show-name>/incoming/<anything>.mp4`
   The filename (without extension) becomes the episode name, e.g.
   `shows/vincenzo/incoming/ep5.mp4` -> `work/vincenzo/ep5/`.
2. Whenever you're ready (drop as many episodes as you like first):
   ```
   venv\Scripts\python.exe src\ingest.py
   ```
   This moves every video waiting in any show's `incoming/` folder into its
   `work/<show>/<episode-name>/source.mp4`, then runs the full 8-stage
   pipeline (extract, separate, transcribe, diarize, translate, TTS, timing
   fit, mix) on that episode **and** on any other episode under `work/`
   that doesn't have a `dubbed.mp4` yet -- so this same command is also how
   you resume an interrupted overnight run, or retry one that failed.

   To run just one already-ingested episode by hand instead:
   ```
   venv\Scripts\python.exe src\run_episode.py work\<show>\<episode-name>
   ```

### How long this takes

Measured on this machine (CPU-only) for the 3-minute, 25-line test clip,
running the whole pipeline from scratch via `run_episode.py`:

| Stage | Time | Rate |
|---|---|---|
| extract | 1s | - |
| separate (Demucs) | 82s | 0.45x real time |
| transcribe (faster-whisper large-v3) | 68s | 0.37x real time |
| diarize (pyannote) | 117s | 0.64x real time |
| translate (Ollama, 1 chunk) | 35s | - |
| TTS (Kokoro, 25 lines) | 26s | - |
| timing fit | 2s | - |
| mix + export | 10s | - |
| **total** | **341s** | **1.85x real time** |

Scaling that up linearly to a full ~60 minute episode (and assuming similar
dialogue density, so roughly 20x as many segments too) suggests somewhere
around **1.5-2 hours total**, with Demucs and diarization as the two
biggest chunks of that. **This is a rough scaled estimate from one short
clip, not a promise** -- a real episode's scene mix (action vs. dialogue)
will shift these numbers, and each run's own timestamped log is the real
answer. It'll also calibrate expectations better for this show's next episode.

### If it crashes or you need to stop it (e.g. overnight)

Just run `src\ingest.py` (or `src\run_episode.py` on that specific episode)
again. Every stage checks what it already finished and picks up from there
-- including transcription, which saves its progress every 20 lines and
resumes from the last completed timestamp rather than starting over. You
will not lose more than a few minutes of work to an interruption, even
mid-transcription.

### After it finishes

Open the review page (see above) to listen through flagged lines, fix
translations, and adjust voices, then hit "Re-run mix" when you're happy.

## Notes on this setup

- Target is CPU-only (per hardware constraints), even though this machine also
  has an NVIDIA RTX 4050 -- CUDA was intentionally skipped to keep things simple
  and match the original plan.
- `requirements.txt` is a frozen `pip freeze` of everything in `venv/`.
