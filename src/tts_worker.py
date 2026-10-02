"""Subprocess worker for a TTS engine (see tts_engine.WorkerEngine).

Usage: python tts_worker.py <engine_name> <target_language>
Protocol: one JSON object per line on stdin; one JSON reply per line on the
real stdout. Library chatter that prints to stdout is redirected to stderr so
it can't corrupt the protocol.
"""
import json
import sys

_proto_out = sys.stdout
sys.stdout = sys.stderr

import soundfile as sf  # noqa: E402
import yaml  # noqa: E402

import pathfix  # noqa: E402,F401
from tts_engine import CONFIG_PATH, load_engine_class  # noqa: E402


def reply(**fields) -> None:
    _proto_out.write(json.dumps(fields) + "\n")
    _proto_out.flush()


def main() -> None:
    name, language = sys.argv[1], sys.argv[2]
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    engine_cfg = config["models"]["tts"]["engines"][name]
    cls = load_engine_class(engine_cfg["class"])
    engine = cls({**config["languages"]["target"][language], "code": language}, engine_cfg.get("settings") or {})
    reply(ok=True, sample_rate=engine.sample_rate)

    for line in sys.stdin:
        request = json.loads(line)
        if request.get("quit"):
            break
        try:
            audio = engine.synthesize(request["text"], request["voice"],
                                      emotion=request.get("emotion"), reference=request.get("reference"),
                                      duration=request.get("duration"))
            sf.write(request["out"], audio, engine.sample_rate)
            reply(ok=True)
        except Exception as e:
            reply(ok=False, error=repr(e))
    engine.close()


if __name__ == "__main__":
    main()
