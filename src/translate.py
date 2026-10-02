"""Stage 5: translate segments to the show's target language using a local LLM via Ollama.

Usage: python src/translate.py work/vincenzo/test_clip [path/to/glossary.json] [source_language] [target_language]
Reads:  <episode_dir>/segments.json
Writes: <episode_dir>/segments.json (adds "target_text" to each segment)

source_language/target_language are codes from config.yaml's languages
allowlist; default to "ko"/"en" when omitted, e.g. for ad-hoc CLI runs.
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

import yaml

import hardware
import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

PROMPT_TEMPLATE = """You are translating dialogue from a {source_name} TV drama into natural, spoken {target_name} for dubbing.

The {source_name} text comes from automatic speech recognition and may contain transcription errors -- infer the intended meaning from context rather than translating nonsense literally. Preserve tone and register (formal/informal, rude/polite). Keep character names and recurring terms consistent across lines.{glossary_block}

Context (surrounding lines, for reference only -- do NOT include these in your output):
{context_block}

Translate ONLY the lines below. For each, aim for a {target_name} line that takes roughly as long to say out loud as its duration (about {words_per_second} words/second of spoken {target_name}), so it's easier to time against the original scene later.

Lines to translate:
{lines_block}

Respond with ONLY valid JSON in this exact form, nothing else:
{{"segments": [{{"id": <id>, "translation": "<translation>"}}, ...]}}
"""


def build_prompt(chunk, context_before, context_after, glossary, source_name, target_name, words_per_second):
    glossary_block = ""
    if glossary:
        terms = ", ".join(f"{k} = {v}" for k, v in glossary.items())
        glossary_block = f"\n\nGlossary (use these exact translations when these terms appear): {terms}"

    def fmt_context(seg):
        target_text = seg.get("target_text")
        return f'id={seg["id"]}: "{seg["source_text"]}"' + (f' -> "{target_text}"' if target_text else "")

    context_lines = [fmt_context(s) for s in (context_before + context_after)]
    context_block = "\n".join(context_lines) if context_lines else "(none)"

    lines_block = "\n".join(
        f'id={seg["id"]} (~{seg["end"] - seg["start"]:.1f}s, target ~'
        f'{max(1, round((seg["end"] - seg["start"]) * words_per_second))} words): "{seg["source_text"]}"'
        for seg in chunk
    )

    return PROMPT_TEMPLATE.format(
        source_name=source_name,
        target_name=target_name,
        glossary_block=glossary_block,
        context_block=context_block,
        words_per_second=words_per_second,
        lines_block=lines_block,
    )


def call_ollama(prompt, model, host):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.3},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"could not reach Ollama at {host} -- is it running? ({e})") from e
    return data["response"]


def validate_and_extract(raw, expected_ids):
    parsed = json.loads(raw)
    items = parsed["segments"] if isinstance(parsed, dict) else parsed
    if not isinstance(items, list):
        raise ValueError("expected a JSON array of segments")

    result = {}
    for item in items:
        seg_id = int(item["id"])
        translation = item["translation"]
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError(f"empty/invalid translation for id {seg_id}")
        result[seg_id] = translation.strip()

    if set(result.keys()) != set(expected_ids):
        raise ValueError(f"expected ids {expected_ids}, got {sorted(result.keys())}")
    return result


def translate_chunk(
    chunk, context_before, context_after, glossary, source_name, target_name, words_per_second,
    model, host, max_retries=3,
):
    expected_ids = [seg["id"] for seg in chunk]
    prompt = build_prompt(chunk, context_before, context_after, glossary, source_name, target_name, words_per_second)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_ollama(prompt, model, host)
            return validate_and_extract(raw, expected_ids)
        except Exception as e:
            last_error = repr(e)
            print(f"  retry {attempt}/{max_retries} for ids {expected_ids}: {last_error}")

    raise RuntimeError(f"translation failed for ids {expected_ids} after {max_retries} attempts: {last_error}")


def translate(
    episode_dir: Path, glossary_path: Path | None = None,
    source_language: str = "ko", target_language: str = "en",
) -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text())
    translation_cfg = config["translation"]
    ollama_cfg = config["models"]["ollama"]
    languages = config["languages"]

    if source_language not in languages["source"]:
        raise ValueError(f"unknown source_language {source_language!r}, expected one of {sorted(languages['source'])}")
    if target_language not in languages["target"]:
        raise ValueError(f"unknown target_language {target_language!r}, expected one of {sorted(languages['target'])}")
    source_name = languages["source"][source_language]["name"]
    target_cfg = languages["target"][target_language]
    target_name = target_cfg["name"]
    words_per_second = target_cfg["words_per_second"]

    glossary = {}
    if glossary_path and glossary_path.exists():
        raw_glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
        glossary = {
            term: entry[target_language]
            for term, entry in raw_glossary.items()
            if target_language in entry
        }

    chunk_size = translation_cfg["chunk_size"]
    context_n = translation_cfg["context_lines"]
    chunk_starts = list(range(0, len(segments), chunk_size))

    for n, start in enumerate(chunk_starts):
        chunk = segments[start:start + chunk_size]
        if all(seg.get("target_text") for seg in chunk):
            continue  # already translated -- resuming after a crash

        context_before = segments[max(0, start - context_n):start]
        context_after = segments[start + chunk_size:start + chunk_size + context_n]

        print(f"translating segments {chunk[0]['id']}-{chunk[-1]['id']} (chunk {n + 1}/{len(chunk_starts)})...")
        translations = translate_chunk(
            chunk, context_before, context_after, glossary, source_name, target_name, words_per_second,
            ollama_cfg["model"], ollama_cfg["host"],
        )
        for seg in chunk:
            seg["target_text"] = translations[seg["id"]]
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    # Don't let Ollama keep its model resident (VRAM) through the next stages.
    hardware.unload_ollama(ollama_cfg["model"], ollama_cfg["host"])
    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    glossary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    source_language = sys.argv[3] if len(sys.argv) > 3 else "ko"
    target_language = sys.argv[4] if len(sys.argv) > 4 else "en"
    result = translate(episode_dir, glossary_path, source_language, target_language)
    print(f"translated {len(result)} segments")
