"""Kokoro-82M: the default engine and the CPU fallback. No emotion/reference support."""
import numpy as np
from kokoro import KPipeline

import hardware
import pathfix  # noqa: F401


class KokoroEngine:
    sample_rate = 24000

    def __init__(self, target_cfg: dict, settings: dict):
        device = hardware.resolve_device(settings.get("device", "auto"))
        self._pipeline = KPipeline(lang_code=target_cfg["kokoro_lang_code"], device=device)

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None) -> np.ndarray:
        chunks = [np.asarray(audio) for _, _, audio in self._pipeline(text, voice=voice)]
        return np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    def close(self) -> None:
        self._pipeline = None
        hardware.release_gpu()
