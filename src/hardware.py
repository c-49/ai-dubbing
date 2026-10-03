"""Hardware detection and GPU memory management shared by every stage.

No stage hardcodes a device: they ask this module. `device: auto` in
config.yaml resolves to CUDA when the installed torch can see an NVIDIA GPU
and to CPU otherwise, so the same code runs on both. `cpu` / `cuda` in the
config force it. The DUBBER_DEVICE environment variable (auto/cpu/cuda)
overrides every stage at once, e.g. to test the CPU path on a GPU machine.

VRAM policy (6GB card): one heavy model on the GPU at a time. Stages call
release_gpu() after dropping their model, and unload_ollama() asks Ollama to
evict its model so it doesn't hold VRAM during the next stage.
"""
import gc
import json
import os
import urllib.request

import pathfix  # noqa: F401


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def resolve_device(requested: str = "auto") -> str:
    """Returns "cuda" or "cpu"."""
    requested = os.environ.get("DUBBER_DEVICE") or requested or "auto"
    if requested == "auto":
        return "cuda" if cuda_available() else "cpu"
    if requested == "cuda" and not cuda_available():
        raise RuntimeError(
            "device is set to 'cuda' but no CUDA GPU is available to torch "
            "(CPU-only environment, or no NVIDIA driver) -- use 'auto' or 'cpu'")
    if requested not in ("cuda", "cpu"):
        raise ValueError(f"unknown device {requested!r}, expected auto, cuda or cpu")
    return requested


def resolve_compute_type(device: str, requested: str = "auto") -> str:
    """faster-whisper/ctranslate2 compute type: float16 on GPU, int8 on CPU."""
    if requested and requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def release_gpu() -> None:
    """Call after dropping references to a model: frees cached VRAM."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_ollama(model: str, host: str) -> None:
    """Asks Ollama to evict the model now (keep_alive=0). Best effort."""
    payload = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        pass


def describe() -> str:
    """One line for logs / check_setup."""
    device = resolve_device("auto")
    if device == "cuda":
        import torch
        props = torch.cuda.get_device_properties(0)
        return f"cuda: {props.name}, {props.total_memory / 2**30:.1f} GiB VRAM"
    return "cpu (no CUDA GPU visible to torch)"
