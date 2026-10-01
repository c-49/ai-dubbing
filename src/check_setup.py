"""
Milestone 0 setup check. Runs each tool once and reports pass/fail.

Does NOT download the large-v3 Whisper model (that happens in Milestone 1) --
it uses the small "base" model here just to prove the mechanics work.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pathfix  # noqa: F401  (patches PATH for ffmpeg/espeak-ng)

WORK_DIR = Path(__file__).resolve().parent.parent / "work" / "_setup_check"
WORK_DIR.mkdir(parents=True, exist_ok=True)

results = {}


def check_ffmpeg():
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    first_line = out.stdout.splitlines()[0] if out.stdout else out.stderr
    return out.returncode == 0, first_line


def check_kokoro():
    from kokoro import KPipeline
    import soundfile as sf

    pipeline = KPipeline(lang_code="a")  # "a" = American English
    audio_path = WORK_DIR / "kokoro_test.wav"
    for _, _, audio in pipeline("Setup test, one two three.", voice="af_bella"):
        sf.write(audio_path, audio, 24000)
        break
    return audio_path.exists(), str(audio_path)


def check_whisper(sample_path: Path):
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(sample_path))
    text = " ".join(seg.text.strip() for seg in segments)
    return bool(text), f"[{info.language}] {text}"


def check_ollama():
    payload = json.dumps({
        "model": "qwen2.5:7b",
        "prompt": "Reply with exactly the word: ok",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return bool(data.get("response")), data.get("response", "").strip()


def check_pyannote():
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if not token_file.exists():
        return None, "No Hugging Face token found yet -- see README for setup steps."
    from pyannote.audio import Pipeline

    Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token_file.read_text().strip(),
    )
    return True, "Pipeline loaded successfully."


def report(name, ok, detail):
    if ok is None:
        status = "SKIP"
    elif ok:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"[{status}] {name}: {detail}")
    results[name] = status


def main():
    ok, detail = check_ffmpeg()
    report("ffmpeg", ok, detail)

    try:
        ok, detail = check_kokoro()
        report("kokoro", ok, detail)
        kokoro_output = WORK_DIR / "kokoro_test.wav"
    except Exception as e:
        report("kokoro", False, repr(e))
        kokoro_output = None

    if kokoro_output and kokoro_output.exists():
        try:
            ok, detail = check_whisper(kokoro_output)
            report("faster-whisper", ok, detail)
        except Exception as e:
            report("faster-whisper", False, repr(e))
    else:
        report("faster-whisper", False, "skipped, no audio sample from kokoro step")

    try:
        ok, detail = check_ollama()
        report("ollama", ok, detail)
    except Exception as e:
        report("ollama", False, repr(e))

    try:
        ok, detail = check_pyannote()
        report("pyannote", ok, detail)
    except Exception as e:
        report("pyannote", False, repr(e))

    print()
    failed = [k for k, v in results.items() if v == "FAIL"]
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("All checks passed (SKIP is fine for pyannote until the HF token is set up).")


if __name__ == "__main__":
    main()
