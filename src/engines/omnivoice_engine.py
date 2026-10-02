"""OmniVoice (k2-fsa, 0.6B): zero-shot voice cloning with explicit duration/speed control.

Runs in its own venv as a worker (see config.yaml). `reference` is the path
of the speaker's reference clip; its transcript is auto-detected by the
model's built-in ASR (cached per reference, so only the first line per
speaker pays for it). Settings: speed (1.0 = model's estimate), num_step.
"""
import numpy as np

import hardware
import pathfix  # noqa: F401


class OmniVoiceEngine:
    def __init__(self, target_cfg: dict, settings: dict):
        import torch
        from omnivoice import OmniVoice

        self._language = target_cfg["code"]
        self._speed = settings.get("speed")
        self._num_step = settings.get("num_step")
        device = hardware.resolve_device(settings.get("device", "auto"))
        dtype = torch.float16 if device == "cuda" else torch.float32
        self._model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=f"{device}:0" if device == "cuda" else device, dtype=dtype)
        self.sample_rate = self._model.sampling_rate
        self._prompts: dict[str, object] = {}

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None) -> np.ndarray:
        kwargs = {}
        if self._speed:
            kwargs["speed"] = self._speed
        if self._num_step:
            kwargs["num_step"] = self._num_step
        if reference:
            reference = str(reference)
            if reference not in self._prompts:
                self._prompts[reference] = self._model.create_voice_clone_prompt(ref_audio=reference)
                # The built-in reference transcriber (Whisper) loads on first use and
                # would otherwise stay resident, pushing a 6GB card into slow spill.
                self._model._asr_pipe = None
                hardware.release_gpu()
            kwargs["voice_clone_prompt"] = self._prompts[reference]
        audio = self._generate(text, kwargs)
        # If the natural reading overruns the line's slot, regenerate with the
        # model's fixed-duration mode so it paces itself to fit.
        if duration and len(audio) / self.sample_rate > duration:
            audio = self._generate(text, {**kwargs, "duration": max(0.3, duration)})
        return audio

    def _generate(self, text, kwargs) -> np.ndarray:
        audio = self._model.generate(text=text, language=self._language, **kwargs)[0]
        return np.asarray(audio, dtype=np.float32).squeeze()

    def close(self) -> None:
        self._model = None
        self._prompts.clear()
        hardware.release_gpu()
