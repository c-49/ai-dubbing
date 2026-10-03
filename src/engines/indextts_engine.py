"""IndexTTS-2.5 (bilibili): zero-shot voice cloning with separate emotion
conditioning and duration control. Source checkout + its own uv env live in
third_party/index-tts (gitignored); this runs as a worker in that env.

`reference` is the speaker's reference clip (timbre). By default the same
clip also sets the emotion; `emotion` may be a path to a different emotion
reference clip. `duration` (slot seconds) is honored by regenerating with
`duration_factor` when the natural reading overruns it.
Settings: use_bf16, emo_alpha.
"""
import sys
from pathlib import Path

import numpy as np

import hardware
import pathfix  # noqa: F401

INDEX_DIR = pathfix.PROJECT_ROOT / "third_party" / "index-tts"
LANG_CODES = {"en": "EN", "es": "ES", "ja": "JA", "zh": "ZH", "ar": "AR"}


class IndexTTSEngine:
    sample_rate = 22050

    def __init__(self, target_cfg: dict, settings: dict):
        sys.path.insert(0, str(INDEX_DIR))
        from indextts.infer_v2_5 import IndexTTS2

        self._lang = LANG_CODES[target_cfg["code"]]
        self._emo_alpha = settings.get("emo_alpha", 1.0)
        device = hardware.resolve_device(settings.get("device", "auto"))
        ckpt = INDEX_DIR / "checkpoints"
        self._model = IndexTTS2(
            cfg_path=str(ckpt / "config.yaml"), model_dir=str(ckpt),
            use_bf16=settings.get("use_bf16", False) and device == "cuda",
            device="cuda:0" if device == "cuda" else "cpu",
        )

    def _infer(self, text, reference, emotion, duration_factor=1.0) -> np.ndarray:
        emo_path = emotion if isinstance(emotion, (str, Path)) else None
        sr, wav = self._model.infer(
            spk_audio_prompt=str(reference), text=text, output_path=None, lang=self._lang,
            emo_audio_prompt=str(emo_path) if emo_path else None, emo_alpha=self._emo_alpha,
            duration_factor=duration_factor, verbose=False,
        )
        self.sample_rate = sr
        return (np.asarray(wav, dtype=np.float32) / 32768.0).reshape(-1)

    def synthesize(self, text, voice, emotion=None, reference=None, duration=None, seed=None) -> np.ndarray:
        if reference is None:
            raise ValueError("IndexTTS needs a reference clip (it only clones)")
        audio = self._infer(text, reference, emotion)
        natural = len(audio) / self.sample_rate
        if duration and natural > duration:
            factor = max(0.5, duration / natural)
            audio = self._infer(text, reference, emotion, duration_factor=factor)
        return audio

    def close(self) -> None:
        self._model = None
        hardware.release_gpu()
