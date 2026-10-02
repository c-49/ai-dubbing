# Upgrade Plan: NVIDIA, Portability, Voice Cloning, Emotional TTS

Companion to `PLAN.md` (the dubbing pipeline), `UI_PLAN.md` (the review app),
and `LANGUAGE_PLAN.md` (multi-language). This file covers four upgrades:
GPU (NVIDIA/CUDA) support, packaging the project so it can be moved to and
installed on any machine without hunting down dependencies, voice cloning,
and emotion-aware TTS.

## Goal
1. Use the NVIDIA GPU where available, and keep a working CPU fallback.
2. Make the project portable: copy the folder to another Windows machine, run
   one setup step, and everything it needs is installed. Updating to new code
   should be easy and never touch models, shows, or work data.
3. Give each original speaker a cloned voice (cross-lingual: Korean/Japanese
   reference, English/Spanish output), so characters sound consistent and the
   3-voice Spanish limit stops mattering.
4. Carry emotion/delivery from the original line into the dub.

## Hard constraints (carried over from PLAN.md)
- **Everything must be free.** No paid APIs or services. Any new model's
  license must allow personal, non-distributed use.
- **Personal, offline use only.** Never distributed or sold.
- **Work one milestone at a time.** Stop after each, show what was built, let
  the user test before moving on.
- **Don't break the CPU path.** A machine with no NVIDIA GPU must still run
  the full pipeline (slower), exactly as today.

## Facts verified (step 0, done)
- `nvidia-smi` confirms an **NVIDIA GeForce RTX 4050 Laptop GPU, driver
  551.76, 6141 MiB VRAM**. The old "AMD" line in `PLAN.md` was wrong and has
  been corrected.
- VRAM is the main planning constraint: 6GB means one heavy model on the GPU
  at a time (see M2). Cloning models that need more than ~5GB need a CPU or
  batch-stage fallback (see M4).

## Decisions proposed (not yet confirmed)
- **No single `.exe`.** CUDA torch alone is several GB, models add 10GB+,
  PyInstaller and torch are fragile together, and a frozen exe makes code
  updates painful. Target instead: a **portable install folder** with a thin
  launcher (`Dubber.bat`, optionally wrapped in a tiny exe later).
- **`uv` for environments.** It can install Python itself, build environments
  from a lockfile, and make extra environments cheap.
- **Code, environment, models, and data are separate.** Updating code
  (`git pull` or unzipping a new `src/`) never touches `models/`, `shows/`,
  or `work/`. If the lockfile changed, the launcher re-syncs the environment.
- **TTS engines run behind an abstraction, new ones in their own venv as a
  subprocess worker.** Cloning models often pin different torch/transformers
  versions than pyannote and Kokoro; isolating them avoids dependency
  conflicts and gives a clean VRAM release between stages.
- **Kokoro stays** as the default engine and the CPU fallback.

## Milestone order
M1 (portability + `uv`/CUDA-or-CPU torch selection skeleton) -> M3 (TTS
abstraction) -> M2 (CUDA + VRAM management) -> M4 -> M5 -> M6 (full
installer/updater). M3 comes before M2 because both touch the same TTS
call sites, and unload/device logic is easier to add once there is one
place that builds engines. Milestone numbers are kept for reference only.

## Milestones

### M1: Portability hygiene
- **Baseline first:** save the current Kokoro output (and timings) on the
  test clip as a reference, so later milestones can check "same output" and
  "no regression" against something real.
- Remove hardcoded machine paths from `pathfix.py` (`C:\Users\reith\...`).
  Bundle ffmpeg (static + shared DLLs) and eSpeak NG into a `bin/` folder
  inside the project and have `pathfix.py` resolve paths relative to the
  project root.
- Point `HF_HOME` (and torch hub cache) at a `models/` folder inside the
  project so downloaded models travel with it.
- Replace the frozen `requirements.txt` with a `pyproject.toml` plus a `uv`
  lockfile. Keep `requirements.txt` only if still useful.
- Make `config.yaml` paths relative to the project root only.
- Start the installer skeleton here: `uv` bootstrap and CUDA-vs-CPU torch
  selection based on detected hardware (full setup/launcher stays in M6).
- Store the Hugging Face token outside the project folder (per-user), so a
  copied folder never carries credentials.
- Bundled ffmpeg/eSpeak NG are fine for personal use; re-check their
  licenses before the folder is ever shared.
- **Done when:** the project folder, copied to a different path (or user
  account), still passes `check_setup.py` with no edits.

### M2: Hardware detection and CUDA across all stages
- Add `src/hardware.py` that resolves `device: auto` (CUDA if available, else
  CPU) and the matching compute type (`float16` on GPU, `int8` on CPU).
  Every stage asks it; no stage hardcodes a device.
- Replace hardcoded `cpu` in `config.yaml` (whisper, demucs, pyannote) and
  `separate.py`'s `--device cpu`. Allow a manual override in config.
- Install the CUDA build of torch/torchaudio. For faster-whisper
  (ctranslate2), supply CUDA 12 + cuDNN 9 via the pip `nvidia-*-cu12`
  packages (and `os.add_dll_directory` in `pathfix.py`) rather than a system
  CUDA install.
- **VRAM management (6GB assumption):** one heavy model on the GPU at a time.
  Add explicit unload after each stage (`del model`, `gc.collect()`,
  `torch.cuda.empty_cache()`). Set Ollama `keep_alive` to 0 after translation
  so it doesn't hold VRAM during TTS. The review UI's cached Kokoro pipelines
  must not compete with a running pipeline stage.
- Re-measure the stage timing table from the README on GPU and record it.
- **Done when:** the test clip runs end-to-end on GPU with no out-of-memory
  errors, the same clip still runs on CPU (forced via config), and the new
  timings are recorded in the README next to the CPU numbers.

### M3: TTS engine abstraction
- Define one interface: `synthesize(text, voice, emotion=None, reference=None)
  -> audio`, plus an engine registry in `config.yaml`
  (`models.tts.engine`, per-engine settings).
- Move current Kokoro logic behind it (`tts.py`, `timing.py`'s re-synthesis,
  and the review UI's `synthesize_clip` all call the same function instead of
  each building their own `KPipeline`).
- Subprocess-worker mode: an engine can declare its own venv and run as a
  worker that loads on demand, serves requests, and exits to free VRAM.
- **Done when:** the Kokoro path works identically through the abstraction
  (same output on the test clip) and a stub second engine can be selected by
  config without changing any caller.

### M4: Voice cloning
- **Research and bake-off first.** Check current candidates (e.g. XTTS-v2 via
  the maintained `coqui-tts` fork, Chatterbox multilingual, CosyVoice 2,
  IndexTTS2) for: license, Korean/Japanese reference tolerance,
  English/Spanish output quality, VRAM use, speed, and duration control.
  This landscape changes quickly; verify specs at implementation time, don't
  trust this list. Also score each on: fits in 6GB VRAM (or has an
  acceptable CPU/batch fallback), and supports emotion-reference
  conditioning (needed by M5). Run the 2-3 best candidates on one real test clip and
  have the user judge by ear. Pick before integrating.
- **Reference clips:** extend `diarize.py` to build a 10-20s reference per
  speaker from clean (non-flagged) vocals, saved per show (e.g.
  `shows/<show>/references/<speaker>.wav`) so a character sounds the same
  across episodes. `voices.json` gains a clone-reference option alongside
  stock voice names.
- **Review UI:** per-speaker choice of stock voice vs. clone, audition and
  swap reference clips (a crying/shouting clip is a bad reference), and
  per-line regenerate.
- **Robustness:** cloned output varies per run and can glitch. Store a seed
  per line, add a retry loop, and add an automatic sanity check (duration
  within bounds; optionally re-transcribe with Whisper and compare). Keep
  `timing.py`'s speed-up/shorten logic.
- **Done when:** the test clip dubs with one cloned voice per speaker, in
  both English and Spanish, and the user judges it acceptable vs. the stock
  Kokoro voices.

### M5: Emotional TTS
- Add an `emotion` field to `segments.json` (editable per line in the review
  UI), filled by one or both of:
  - a local speech-emotion classifier run on each original line's separated
    vocals (free, cheap), and
  - the translation LLM, which can tag delivery from context (angry,
    whispering, sarcastic, etc.).
- Each engine maps the field to its own control: tags, an intensity value, or
  using the original line's audio as an **emotion reference** (most
  dubbing-appropriate since it carries delivery, not just a label).
- Test whether emotion-reference conditioning leaks the source language's
  accent into the target language; fall back to labels/intensity if so.
- **Done when:** a test scene with clearly different emotions (calm, angry,
  crying) sounds audibly different in the dub, judged by ear, with no
  regression in timing fit.

### M6: Launcher, installer, updater
- `setup.bat` (or `setup.ps1`): installs `uv`, creates the environment from
  the lockfile, picking the CUDA or CPU torch build based on detected
  hardware, creates extra venvs for subprocess engines, and installs Ollama
  via winget and pulls the configured model.
- `Dubber.bat`: ensures the environment matches the lockfile (auto `uv sync`
  if it changed), starts the Flask server, opens the browser.
- Updating code: `git pull` or a zip swap replaces `src/` and `review_ui/`;
  `models/`, `shows/`, and `work/` are never touched. Optional
  `update.bat` wraps this.
- Optional `make_bundle` script that zips the environment and models for a
  fully offline move to another machine.
- First-run prompts for what can't be bundled: the Hugging Face token for
  pyannote's gated models (once per machine).
- Optional later: wrap `Dubber.bat` in a tiny exe for a nicer double-click.
- **Done when:** on a second Windows machine (or a clean user account) with
  nothing installed, running setup then the launcher gets to the review
  page and dubs the test clip, and a code update applied afterwards leaves
  shows, voices, and work data intact.

## Open items to revisit during implementation
- Confirm GPU model and VRAM (see "Facts to verify first"); several choices
  (which cloning models fit, whether Ollama and TTS can share VRAM) depend on
  it.
- Which cloning model(s) to adopt, decided by the M4 bake-off, not up front.
- Whether `qwen2.5:7b` should stay the translation model now that more VRAM
  headroom is managed explicitly (carried over from LANGUAGE_PLAN.md).
- Whether pyannote and faster-whisper on CUDA behave on this Windows setup
  without further DLL workarounds (`pathfix.py` already needed some for
  torchcodec).
- Whether a hand-written `words_per_second` still makes sense once cloned
  voices (with different pacing than Kokoro) are in use. Likely needs to be
  per-engine, not just per-language.
- Licensing re-check for every new model, even for personal use.
- Machines without NVIDIA: confirm the CPU build and the Kokoro-only path
  stay supported on a clean install.

## M4 research notes (verified Oct 2026 against project repos)

Constraints applied: free, personal use, fits a 6GB card (alone on the GPU),
Korean/Japanese reference -> English/Spanish output.

| Model | Size / VRAM | License | Strengths | Concerns |
|---|---|---|---|---|
| **Chatterbox Multilingual** (Resemble AI) | 500M; ~3.5GB | MIT | 23 languages incl. ko/ja/es/en; `exaggeration` + `cfg_weight` emotion knobs; simple pip install | Docs warn the output inherits the reference's accent (mitigate with `cfg_weight=0`); no duration control; no emotion-from-audio; watermarks output (Perth) |
| **OmniVoice** (k2-fsa) | 0.6B; fast (RTF ~0.025) | Apache-2.0 per GitHub, CC-BY-NC per HF card (fine for personal use either way) | 600+ languages; explicit `duration` and `speed` control (ideal for timing fit); non-verbal tags, whisper style | Docs also warn of reference-language accent; emotion control is limited |
| **IndexTTS-2.5** (bilibili) | ~6GB reported, tight | Custom bilibili license (ok for personal use) | Emotion from a separate reference audio or 8-float emotion vector (best fit for M5); `duration_factor` | Tightest on VRAM; Korean not listed as a supported language (English/Spanish/Japanese are, output side is what matters); heavier Windows install (DeepSpeed optional) |

Considered and dropped: VoxCPM2 (needs ~8GB), XTTS-v2 (older, non-commercial
CPML license, weaker than the above; kept as a fallback if all three fail),
Fish Speech (CC-BY-NC weights, no clear advantage here).

Bake-off plan: for each of the three, build a per-speaker reference from the
test clip's separated vocals (`speaker_samples/`), dub the same 25 English
lines (plus a Spanish sample), run them through the normal timing-fit + mix,
and compare the resulting `dubbed.mp4` files by ear against Kokoro. Each
candidate gets its own venv under `.venv-engines/`, run through the M3
subprocess-worker interface.
