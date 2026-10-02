"""Chatterbox Multilingual (Resemble AI, MIT): zero-shot voice cloning.

Runs in its own venv as a worker (see config.yaml). `reference` is the path of
the speaker's reference clip; `voice` is unused for cloning. Settings:
  exaggeration (0-1, emotion intensity), cfg_weight (0 reduces accent bleed
  from a foreign-language reference), temperature, seed.
"""
import numpy as np
import torch

import hardware
import pathfix  # noqa: F401


class ChatterboxEngine:
    def __init__(self, target_cfg: dict, settings: dict):
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        self._language = target_cfg["code"]
        self._exaggeration = settings.get("exaggeration", 0.5)
        self._cfg_weight = settings.get("cfg_weight", 0.5)
        self._temperature = settings.get("temperature", 0.8)
        self._seed = settings.get("seed")
        self._device = hardware.resolve_device(settings.get("device", "auto"))
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=self._device)
        self.sample_rate = self._model.sr

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None, seed=None) -> np.ndarray:
        seed = seed if seed is not None else self._seed
        if seed is not None:
            torch.manual_seed(seed)
        exaggeration = float(emotion) if isinstance(emotion, (int, float)) else self._exaggeration
        wav = self._model.generate(
            text, language_id=self._language,
            audio_prompt_path=str(reference) if reference else None,
            exaggeration=exaggeration, cfg_weight=self._cfg_weight, temperature=self._temperature,
        )
        return wav.squeeze().cpu().numpy().astype(np.float32)

    def close(self) -> None:
        self._model = None
        hardware.release_gpu()
