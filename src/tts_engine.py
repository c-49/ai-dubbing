"""TTS engine abstraction: one interface every caller (tts.py, timing.py's
re-synthesis, the review UI) goes through, so engines can be swapped by
config (models.tts.engine) without touching any caller.

An engine is a class with:
    sample_rate: int
    __init__(target_cfg: dict, settings: dict)   # languages.target.<code> (+ "code"), models.tts.engines.<name>.settings
    synthesize(text, voice, emotion=None, reference=None, duration=None, seed=None) -> np.ndarray (mono float32)
    close() -> None                               # free VRAM/RAM

`emotion`, `reference` (path to a reference clip), `duration` (seconds the
line should fit in, i.e. its time slot) and `seed` (makes a sampled take
reproducible; retries use a different one) are optional hints; an engine that
doesn't support them ignores them. timing.py still speeds up / shortens
whatever doesn't fit afterwards.

Engines are registered in config.yaml under models.tts.engines.<name>:
    class: "engines.kokoro_engine:KokoroEngine"
    worker:                      # optional -- run in a separate process
      python: path/to/venv/python.exe   # that engine's own venv (relative paths are project-relative)
A worker engine is spawned on first use and exits on release_engines(), so
engines with conflicting dependencies are isolated and VRAM is freed cleanly.
"""
import importlib
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import pathfix  # noqa: F401

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
WORKER_SCRIPT = Path(__file__).resolve().parent / "tts_worker.py"

_engines: dict[tuple[str, str], object] = {}
_lock = threading.Lock()


def load_engine_class(spec: str):
    module_name, _, class_name = spec.partition(":")
    return getattr(importlib.import_module(module_name), class_name)


class WorkerEngine:
    """Proxy that runs an engine in a subprocess (its own venv) and talks to
    it over a line-based JSON protocol on stdin/stdout, passing audio through
    temporary wav files."""

    MAX_ATTEMPTS = 3

    def __init__(self, name: str, language: str, python: str):
        self._lock = threading.Lock()
        self._python_path = Path(python)
        if not self._python_path.is_absolute():
            self._python_path = ROOT_DIR / self._python_path
        self._name, self._language = name, language
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [str(self._python_path), str(WORKER_SCRIPT), self._name, self._language],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.sample_rate = self._read_reply()["sample_rate"]

    def _restart(self) -> None:
        # A CUDA error inside the worker poisons its whole context, so a retry
        # needs a fresh process, not just another request.
        self._proc.kill()
        self._proc.wait()
        self._start()

    def _read_reply(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("TTS worker exited unexpectedly (see its stderr above)")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"TTS worker error: {reply.get('error')}")
        return reply

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None, seed=None) -> np.ndarray:
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            request = {"text": text, "voice": voice, "emotion": emotion,
                       "reference": str(reference) if reference else None, "duration": duration,
                       "seed": seed, "out": str(out)}
            # Cloning models fail stochastically (invalid sampled tokens, crashes):
            # restart the worker and retry, since sampling differs each attempt.
            for attempt in range(1, self.MAX_ATTEMPTS + 1):
                try:
                    self._proc.stdin.write(json.dumps(request) + "\n")
                    self._proc.stdin.flush()
                    self._read_reply()
                    break
                except (RuntimeError, OSError) as e:
                    if attempt == self.MAX_ATTEMPTS:
                        raise
                    print(f"  TTS worker failed ({e}); restarting and retrying "
                          f"({attempt}/{self.MAX_ATTEMPTS - 1})", flush=True)
                    self._restart()
            audio, _ = sf.read(out, dtype="float32")
            return audio

    def close(self) -> None:
        try:
            self._proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()


def get_engine(target_language: str, engine_name: str | None = None):
    """Returns the (cached) engine for this target language. engine_name
    defaults to config.yaml's models.tts.engine."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tts_cfg = config["models"]["tts"]
    # DUBBER_TTS_ENGINE overrides config.yaml (used by the bake-off script).
    name = engine_name or os.environ.get("DUBBER_TTS_ENGINE") or tts_cfg["engine"]
    key = (name, target_language)
    with _lock:
        if key not in _engines:
            engine_cfg = tts_cfg["engines"][name]
            worker = engine_cfg.get("worker")
            if worker:
                _engines[key] = WorkerEngine(name, target_language, worker["python"])
            else:
                cls = load_engine_class(engine_cfg["class"])
                target_cfg = {**config["languages"]["target"][target_language], "code": target_language}
                _engines[key] = cls(target_cfg, engine_cfg.get("settings") or {})
        return _engines[key]


def release_engines() -> None:
    """Closes every loaded engine (frees VRAM/RAM; workers exit)."""
    with _lock:
        for engine in _engines.values():
            engine.close()
        _engines.clear()


def synthesize(text: str, voice: str, target_language: str, emotion=None, reference=None,
               duration=None, seed=None, engine_name: str | None = None) -> tuple[np.ndarray, int]:
    """Convenience: returns (audio, sample_rate)."""
    engine = get_engine(target_language, engine_name)
    return engine.synthesize(text, voice, emotion=emotion, reference=reference, duration=duration,
                             seed=seed), engine.sample_rate


# --- Per-segment synthesis: stock vs. cloned voice, with sanity-checked retries ---

CLONE_VOICE = "clone"     # a segment/speaker voice of "clone" means: clone the original speaker
MAX_TAKES = 3             # cloned takes tried per line before keeping the best one
NEUTRAL_EMOTIONS = {None, "", "neutral"}
MIN_LEVEL_DB = -45.0       # a take quieter than this is a dead/silent generation (seen: -86 dB)
NORMAL_LEVEL_DB = -21.5    # typical level of a cloned line (90th-percentile frame RMS)
LEVEL_FOLLOW = 0.5         # a delivery-referenced line keeps this share of the original's loudness offset...
LEVEL_FOLLOW_MAX_DB = 3.0  # ...capped, so soft lines stay audible and loud ones don't clip
MAX_GAIN_DB = 12.0


def level_db(audio: np.ndarray, sample_rate: int) -> float:
    """Loud-part level of a clip: 90th-percentile of 25 ms frame RMS, in dB."""
    frame = max(1, round(0.025 * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return -120.0
    frames = audio[: n * frame].reshape(n, frame)
    return float(np.percentile(20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-8), 90))


def clone_engine_name() -> str:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config["models"]["tts"].get("clone_engine", "omnivoice")


MIN_CAP_SECONDS = 3.0     # even one-word lines may take this long (the model's own minimum is ~2.2s)


def duration_bounds(text: str, words_per_second: float) -> tuple[float, float]:
    """(shortest, longest) plausible length in seconds for this text. Cloned
    takes can come out near-empty (seen: 0.12s for a sentence) or runaway
    (seen: ~12s for seven words, identical across seeds), so the longest is
    also handed to the engine as a hard cap. Bounds are deliberately wide."""
    expected = max(1, len(text.split())) / words_per_second
    return 0.3 * expected - 0.1, max(2.5 * expected + 1.0, MIN_CAP_SECONDS)


def synthesize_segment(seg: dict, text: str, target_language: str, duration=None) -> tuple[np.ndarray, int]:
    """Synthesizes `text` for a segment with its voice: a Kokoro-style stock
    voice, or (voice == "clone") the clone engine using seg["reference"].
    Cloned takes are re-rolled with a new seed if they look glitched, and the
    seed that was kept is stored in seg["seed"]. Returns (audio, sample_rate)."""
    reference = seg.get("reference")
    reference = str(ROOT_DIR / reference) if reference else None

    if seg.get("voice") != CLONE_VOICE:
        engine = get_engine(target_language)
        audio = engine.synthesize(text, seg["voice"], emotion=seg.get("emotion"),
                                  reference=reference, duration=duration)
        return audio, engine.sample_rate

    if not reference:
        raise ValueError(f"segment {seg.get('id')} uses a cloned voice but has no reference clip")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    words_per_second = config["languages"]["target"][target_language]["words_per_second"]
    engine = get_engine(target_language, clone_engine_name())

    # Delivery: for an emotional line, clone from the line's OWN original audio
    # (the model copies how it was said), not the speaker's usual neutral reference.
    emotion = seg.get("emotion")
    if (config.get("emotion", {}).get("use_line_reference", True) and emotion not in NEUTRAL_EMOTIONS
            and seg.get("line_reference") and (ROOT_DIR / seg["line_reference"]).exists()):
        reference = str(ROOT_DIR / seg["line_reference"])

    own_audio_reference = reference != (str(ROOT_DIR / seg["reference"]))
    shortest, longest = duration_bounds(text, words_per_second)
    hint = min(duration, longest) if duration else longest   # slot, but never beyond a plausible length
    base_seed = seg.get("seed_base", seg["id"] * 100)
    best = None
    for take in range(MAX_TAKES + (1 if own_audio_reference else 0)):
        if own_audio_reference and take == MAX_TAKES:
            # The line's own audio keeps giving bad takes (e.g. a near-silent original):
            # fall back to the speaker's usual reference for a last attempt.
            reference, own_audio_reference = str(ROOT_DIR / seg["reference"]), False
            print(f"  id={seg.get('id')}: delivery reference failed; using the speaker's usual reference", flush=True)
        seed = base_seed + take
        audio = engine.synthesize(text, seg["voice"], emotion=seg.get("emotion"), reference=reference,
                                  duration=hint, seed=seed)
        seconds = len(audio) / engine.sample_rate
        ok = shortest <= seconds <= longest * 1.1 and level_db(audio, engine.sample_rate) >= MIN_LEVEL_DB
        if best is None or ok:
            best = (audio, seed, own_audio_reference)
        if ok:
            break
        print(f"  id={seg.get('id')}: take {take + 1} looks glitched ({seconds:.2f}s for {text!r}); retrying", flush=True)
    seg["seed"] = best[1]
    audio = best[0]
    if best[2]:
        # Copying a quiet or shouted original also copies its absolute level. Keep the
        # contrast but bring it near a normal level so every line stays audible.
        offset = max(-LEVEL_FOLLOW_MAX_DB, min(LEVEL_FOLLOW_MAX_DB, LEVEL_FOLLOW * seg.get("loudness_offset_db", 0.0)))
        gain = max(-MAX_GAIN_DB, min(MAX_GAIN_DB, NORMAL_LEVEL_DB + offset - level_db(audio, engine.sample_rate)))
        audio = np.clip(audio * (10 ** (gain / 20)), -1.0, 1.0).astype(np.float32)
    return audio, engine.sample_rate
