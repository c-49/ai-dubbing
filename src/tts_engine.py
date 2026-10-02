"""TTS engine abstraction: one interface every caller (tts.py, timing.py's
re-synthesis, the review UI) goes through, so engines can be swapped by
config (models.tts.engine) without touching any caller.

An engine is a class with:
    sample_rate: int
    __init__(target_cfg: dict, settings: dict)   # languages.target.<code>, models.tts.engines.<name>.settings
    synthesize(text, voice, emotion=None, reference=None) -> np.ndarray (mono float32)
    close() -> None                               # free VRAM/RAM

`emotion` and `reference` (path to a reference clip) are optional hints; an
engine that doesn't support them ignores them.

Engines are registered in config.yaml under models.tts.engines.<name>:
    class: "engines.kokoro_engine:KokoroEngine"
    worker:                      # optional -- run in a separate process
      python: path/to/venv/python.exe   # that engine's own venv (relative paths are project-relative)
A worker engine is spawned on first use and exits on release_engines(), so
engines with conflicting dependencies are isolated and VRAM is freed cleanly.
"""
import importlib
import json
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

    def __init__(self, name: str, language: str, python: str):
        self._lock = threading.Lock()
        python_path = Path(python)
        if not python_path.is_absolute():
            python_path = ROOT_DIR / python_path
        self._proc = subprocess.Popen(
            [str(python_path), str(WORKER_SCRIPT), name, language],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
        )
        self.sample_rate = self._read_reply()["sample_rate"]

    def _read_reply(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("TTS worker exited unexpectedly (see its stderr above)")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"TTS worker error: {reply.get('error')}")
        return reply

    def synthesize(self, text, voice, emotion=None, reference=None) -> np.ndarray:
        with self._lock, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.wav"
            request = {"text": text, "voice": voice, "emotion": emotion,
                       "reference": str(reference) if reference else None, "out": str(out)}
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            self._read_reply()
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
    name = engine_name or tts_cfg["engine"]
    key = (name, target_language)
    with _lock:
        if key not in _engines:
            engine_cfg = tts_cfg["engines"][name]
            worker = engine_cfg.get("worker")
            if worker:
                _engines[key] = WorkerEngine(name, target_language, worker["python"])
            else:
                cls = load_engine_class(engine_cfg["class"])
                target_cfg = config["languages"]["target"][target_language]
                _engines[key] = cls(target_cfg, engine_cfg.get("settings") or {})
        return _engines[key]


def release_engines() -> None:
    """Closes every loaded engine (frees VRAM/RAM; workers exit)."""
    with _lock:
        for engine in _engines.values():
            engine.close()
        _engines.clear()


def synthesize(text: str, voice: str, target_language: str, emotion=None, reference=None,
               engine_name: str | None = None) -> tuple[np.ndarray, int]:
    """Convenience: returns (audio, sample_rate)."""
    engine = get_engine(target_language, engine_name)
    return engine.synthesize(text, voice, emotion=emotion, reference=reference), engine.sample_rate
