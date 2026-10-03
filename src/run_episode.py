"""Runs all pipeline stages for one episode, in order, skipping finished work.

Usage: python src/run_episode.py work/<show>/<episode_name>

Safe to re-run after a crash or Ctrl+C -- each stage figures out on its own
what (if anything) is already done and picks up from there (see the module
docstrings in extract.py/separate.py/transcribe.py/etc. for how each one
decides what's left to do). The show name is the episode's parent folder,
e.g. work/vincenzo/ep5 -> show "vincenzo" -> shows/vincenzo/glossary.json,
voices.json, and config.json (source_language/target_language) are used
automatically if present; source_language defaults to "ko" if config.json
is missing.

Also maintains, for the review web app's benefit:
- <episode_dir>/status.json -- status (running/done/failed), which stage is
  running (e.g. "3/8 transcribe (faster-whisper)"), and the error on failure.
- <episode_dir>/run.log -- a timestamped stage-by-stage log, appended across
  runs/resumes (not each stage's own tool output -- ffmpeg/Demucs/Whisper
  console spam still only goes to this process's terminal).

And takes a global lock (work/.pipeline.lock) for the duration of the run so
the web app's background worker and a manually-run ingest.py never process
two episodes at once -- the pipeline's RAM budget assumes only one heavy
stage (Whisper, the LLM, etc.) is loaded at a time.
"""
import json
import os
import sys
import time
from pathlib import Path

import extract
import separate
import transcribe
import diarize
import emotion
import translate
import tts
import timing
import mix
import tts_engine

ROOT_DIR = Path(__file__).resolve().parent.parent
SHOWS_DIR = ROOT_DIR / "shows"
LOCK_PATH = ROOT_DIR / "work" / ".pipeline.lock"
LOCK_POLL_SECONDS = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class EpisodeLogger:
    """Writes timestamped lines to both the console and <episode_dir>/run.log."""

    def __init__(self, episode_dir: Path):
        self._file = open(episode_dir / "run.log", "a", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def run_stage(logger: EpisodeLogger, name: str, fn, *args) -> None:
    logger.log(f"--- {name} ---")
    t0 = time.time()
    fn(*args)
    logger.log(f"--- {name} done in {time.time() - t0:.0f}s ---")


def _update_status(episode_dir: Path, **fields) -> None:
    path = episode_dir / "status.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.update(fields)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _acquire_lock(label: str, log_fn=log) -> None:
    """Blocks until no other episode is running. If this looks stuck, check
    whether something is actually running; a process killed without
    cleanup can leave work/.pipeline.lock behind -- delete it by hand."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    waited = False
    while True:
        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, label.encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            if not waited:
                log_fn(f"waiting for another episode to finish (lock held by {LOCK_PATH.read_text(encoding='utf-8').strip()})...")
                waited = True
            time.sleep(LOCK_POLL_SECONDS)


def _release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def run(episode_dir: Path) -> None:
    show = episode_dir.resolve().parent.name
    label = f"{show}/{episode_dir.name}"

    glossary_path = SHOWS_DIR / show / "glossary.json"
    voices_path = SHOWS_DIR / show / "voices.json"
    config_path = SHOWS_DIR / show / "config.json"
    glossary_path = glossary_path if glossary_path.exists() else None
    voices_path = voices_path if voices_path.exists() else None

    glossary_terms = None
    if glossary_path:
        glossary_terms = list(json.loads(glossary_path.read_text(encoding="utf-8")).keys())

    show_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    source_language = show_config.get("source_language", "ko")
    target_language = show_config.get("target_language", "en")

    stages = [
        ("extract audio", extract.extract_audio, (episode_dir,)),
        ("separate vocals (Demucs)", separate.separate, (episode_dir,)),
        ("transcribe (faster-whisper)", transcribe.transcribe, (episode_dir, glossary_terms, source_language)),
        ("diarize (pyannote)", diarize.diarize, (episode_dir,)),
        ("delivery analysis", emotion.analyze, (episode_dir,)),
        ("translate (Ollama)", translate.translate, (episode_dir, glossary_path, source_language, target_language)),
        ("TTS", tts.synthesize, (episode_dir, voices_path, target_language)),
        ("timing fit", timing.fit_timing, (episode_dir, source_language, target_language)),
        ("mix + export", mix.mix, (episode_dir, source_language, target_language)),
    ]

    logger = EpisodeLogger(episode_dir)
    try:
        _acquire_lock(label, log_fn=logger.log)
        try:
            _update_status(episode_dir, status="running", error=None,
                            stage=None, stage_num=0, stage_total=len(stages))

            t_start = time.time()
            logger.log(f"Starting episode: {label}")

            for i, (name, fn, args) in enumerate(stages, start=1):
                stage_label = f"{i}/{len(stages)} {name}"
                _update_status(episode_dir, stage=stage_label, stage_num=i)
                run_stage(logger, stage_label, fn, *args)

            logger.log(f"Episode complete in {time.time() - t_start:.0f}s total -- see {episode_dir / 'dubbed.mp4'}")
            _update_status(episode_dir, status="done", error=None)
        except BaseException as e:
            logger.log(f"FAILED: {e!r}")
            _update_status(episode_dir, status="failed", error=str(e) or type(e).__name__)
            raise
        finally:
            tts_engine.release_engines()  # free VRAM/RAM held by the TTS engine
            _release_lock()
    finally:
        logger.close()


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    try:
        run(episode_dir)
    except KeyboardInterrupt:
        log("Interrupted -- safe to resume later by running this same command again.")
        sys.exit(1)
