"""Stub engine for testing the abstraction: a sine tone whose length scales
with the text. Selecting it by config proves callers don't depend on Kokoro."""
import numpy as np


class StubEngine:
    sample_rate = 16000

    def __init__(self, target_cfg: dict, settings: dict):
        self._seconds_per_char = settings.get("seconds_per_char", 0.05)

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None, seed=None) -> np.ndarray:
        n = int(self.sample_rate * self._seconds_per_char * max(1, len(text)))
        t = np.arange(n) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def close(self) -> None:
        pass
