# Kdrama English Dubbing Pipeline

See `PLAN.md` for the full project plan. This file only covers day-to-day setup
and running instructions, updated as milestones land.

## Quick start (any Windows 10/11 machine)

1. Copy this folder to the machine (anywhere; no need to keep the same path).
   You do **not** need to copy `.venv*`, `bin/` or `work/`; `models/` is
   optional (copy it to skip re-downloading ~5GB of models).
2. Double-click **`setup.bat`**. It installs what is missing: `uv` (which
   fetches Python itself), the Python packages (CUDA build of torch if an
   NVIDIA GPU is detected, else CPU), ffmpeg and eSpeak NG (bundled into
   `bin/`), the voice-cloning engine's own environment, Ollama and the
   translation model. It asks once for your Hugging Face token (steps are
   printed; see below) and finishes by running `src/check_setup.py`.
   Options: `-Cpu` (force CPU), `-NoClone`, `-SkipOllama`. It is safe to
   re-run if anything fails part-way.
3. Double-click **`Dubber.bat`** any time to start the review app. It
   re-syncs the environment if the code was updated, starts Ollama if needed,
   starts the server and opens the browser. (Run it first and it will run
   setup for you.)
4. **`update.bat`** pulls new code (git checkout) and re-syncs the
   environment. If you got the project as a zip, copy the new files over the
   old ones (keep `models/`, `shows/`, `work/`) and run `update.bat`.
   Code, environments, models and your data are separate, so updating never
   touches `models/`, `shows/` or `work/`.

Needs: Windows with `winget` (built into Windows 11 / current Windows 10),
internet for the first setup, and ~25GB free disk. Verified by copying the
project to a new path without any environment and running setup from scratch.

## Setup details (Milestone 0 -- historical, `setup.bat` now does all of this)

Already done by the assistant on this machine:
- Python 3.11 installed (alongside whatever other Python versions you have), via winget.
- ffmpeg installed via winget (`ffmpeg`, `ffprobe` on PATH).
- eSpeak NG installed via winget (Kokoro TTS needs it on Windows).
- Python environment lives in `.venv/`, built by `uv` from `pyproject.toml` + `uv.lock` (Python 3.11).
- Installed into `.venv/`: PyTorch (CPU build by default; `uv sync --extra cuda` for NVIDIA), faster-whisper, demucs, pyannote.audio, kokoro.
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
   .venv\Scripts\huggingface-cli login
   ```
   Paste the token when prompted. This writes the token to
   `~/.cache/huggingface/token`, which `check_setup.py` and the diarization
   script look for.

You don't need this for Milestones 0-3. It's only required starting Milestone 4.

## Running the setup check

```
.venv\Scripts\python.exe src\check_setup.py
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
.venv\Scripts\python.exe review_ui\app.py
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
   .venv\Scripts\python.exe src\ingest.py
   ```
   This moves every video waiting in any show's `incoming/` folder into its
   `work/<show>/<episode-name>/source.mp4`, then runs the full 8-stage
   pipeline (extract, separate, transcribe, diarize, translate, TTS, timing
   fit, mix) on that episode **and** on any other episode under `work/`
   that doesn't have a `dubbed.mp4` yet -- so this same command is also how
   you resume an interrupted overnight run, or retry one that failed.

   To run just one already-ingested episode by hand instead:
   ```
   .venv\Scripts\python.exe src\run_episode.py work\<show>\<episode-name>
   ```

### How long this takes

Measured on this machine (RTX 4050 Laptop, 6GB) for the 3-minute, 25-line
test clip, running the whole pipeline from scratch via `run_episode.py`.
The CPU column is a re-run on the current code/environment (`.venv`); the
original first-ever CPU measurement was 341s total, so expect some
run-to-run and machine-load variation (+/- 30%):

| Stage | GPU (`.venv-cuda`) | CPU (`.venv`) |
|---|---|---|
| extract | 1s | 0s |
| separate (Demucs) | 22s | 91s |
| transcribe (faster-whisper large-v3) | 33s | 84s |
| diarize (pyannote) | 26s | 169s |
| translate (Ollama, 1 chunk) | 45s | 37s |
| TTS (Kokoro, 25 lines) | 33s | 43s |
| timing fit | 17s | 18s |
| mix + export | 12s | 12s |
| **total** | **188s (1.0x real time)** | **456s (2.5x real time)** |

Peak VRAM on the GPU run was about 5.0 GB of 6 GB (Ollama is the biggest
single consumer, ~4.2 GB; each stage frees its model before the next one
loads). Ollama, TTS and the pipeline's other stages use the GPU
automatically when available. Set `device: cpu` per model in `config.yaml`,
or set the environment variable `DUBBER_DEVICE=cpu`, to force CPU.
GPU and CPU runs transcribe slightly differently (float16 vs int8), so the
flagged-line count can differ between them.

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

## Voice cloning (OmniVoice)

Besides the stock Kokoro voices, each speaker can be dubbed with a clone of
their own original voice (Korean/Japanese reference -> English/Spanish
output), so characters sound consistent and Spanish is no longer limited to
3 voices.

- In the review page, pick **Clone original voice** in a speaker's voice
  dropdown (or in `shows/<show>/voices.json`, set the speaker to `"clone"`).
  The speaker card then shows a **reference clip** picker: "auto" joins that
  speaker's cleanest lines (~15s); you can instead pick one of their sample
  clips (avoid a shouting/crying clip) and audition it with "play reference".
  A clip you pick is saved to `shows/<show>/references/<speaker>.wav` and used
  for every episode of that show.
- Each line has a **Redo** button: cloned takes vary, so re-roll one that
  sounds off. Takes that look glitched (far too short/long for the text) are
  re-rolled automatically, and the seed used is saved in `segments.json`.
- The clone engine (`models.tts.clone_engine` in `config.yaml`) runs as its
  own worker process in `.venv-engines/omnivoice` and exits when done, so it
  never shares the 6GB GPU with Ollama. Cloning a 25-line clip takes ~4 min
  (about 10s per line); stock Kokoro voices stay much faster.
- Accent: cloning from a Korean voice tends to leave a slight accent. A
  listening test picked `guidance_scale: 1.0` (the model default of 2.0
  pulls harder toward the reference accent); change it under
  `models.tts.engines.omnivoice.settings`.
- Cloned and stock speakers can be mixed in one episode.

### Delivery (emotion) in cloned dubs

A pipeline stage ("delivery analysis", `src/emotion.py`) measures each original
line's loudness and pitch against that speaker's own usual level and labels it
`neutral`, `intense` or `soft` (it measures arousal only, so it can't tell anger
from joy; lines under 1s are left neutral). The **Delivery** column in the
review page shows and edits the label (also: angry, sad, crying, whisper,
happy). For a cloned speaker, a non-neutral line is cloned from *that line's
own original audio* instead of the speaker's usual reference, so the dub
inherits how the actor said it. `whisper` also switches on OmniVoice's
whisper style. Levels are kept near normal so soft lines stay audible. Your
manual labels are never overwritten. Turn this off with
`emotion.use_line_reference: false` in `config.yaml`. Stock Kokoro voices
have no emotion control and ignore the label.

Setting up the engine on a new machine: `uv venv .venv-engines/omnivoice
--python 3.11`, install torch 2.8.0 (CUDA build from
https://download.pytorch.org/whl/cu126), then `omnivoice soundfile pyyaml
numpy`. (M6's setup script will automate this.)

## Notes on this setup

- Runs on an NVIDIA GPU when the CUDA environment is installed (`uv sync --extra cuda`),
  and on CPU otherwise (`uv sync --extra cpu`) -- same code either way.
- Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`. Recreate the environment with `uv sync --extra cpu` (or `--extra cuda`). The old `requirements.txt`/`venv/` are legacy and can be deleted once you are happy with `.venv/`.
- The project is self-contained: ffmpeg and eSpeak NG live in `bin/`, downloaded models in `models/` (the Hugging Face token stays per-user in `~/.cache/huggingface/token`).
