"""Stage 5: translate Korean segments to English using a local LLM via Ollama.

Usage: python src/translate.py work/vincenzo/test_clip [path/to/glossary.json]
Reads:  <episode_dir>/segments.json
Writes: <episode_dir>/segments.json (adds "english" to each segment)
"""
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

import yaml

import pathfix  # noqa: F401

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

WORDS_PER_SECOND = 2.3  # rough average spoken English pace, for length guidance

PROMPT_TEMPLATE = """You are translating dialogue from a Korean TV drama into natural, spoken English for dubbing.

The Korean text comes from automatic speech recognition and may contain transcription errors -- infer the intended meaning from context rather than translating nonsense literally. Preserve tone and register (formal/informal, rude/polite). Keep character names and recurring terms consistent across lines.{glossary_block}

Context (surrounding lines, for reference only -- do NOT include these in your output):
{context_block}

Translate ONLY the lines below. For each, aim for an English line that takes roughly as long to say out loud as its duration (about {words_per_second} words/second of spoken English), so it's easier to time against the original scene later.

Lines to translate:
{lines_block}

Respond with ONLY valid JSON in this exact form, nothing else:
{{"segments": [{{"id": <id>, "english": "<translation>"}}, ...]}}
"""


def build_prompt(chunk, context_before, context_after, glossary):
    glossary_block = ""
    if glossary:
        terms = ", ".join(f"{k} = {v}" for k, v in glossary.items())
        glossary_block = f"\n\nGlossary (use these exact translations when these terms appear): {terms}"

    def fmt_context(seg):
        english = seg.get("english")
        return f'id={seg["id"]}: "{seg["korean"]}"' + (f' -> "{english}"' if english else "")

    context_lines = [fmt_context(s) for s in (context_before + context_after)]
    context_block = "\n".join(context_lines) if context_lines else "(none)"

    lines_block = "\n".join(
        f'id={seg["id"]} (~{seg["end"] - seg["start"]:.1f}s, target ~'
        f'{max(1, round((seg["end"] - seg["start"]) * WORDS_PER_SECOND))} words): "{seg["korean"]}"'
        for seg in chunk
    )

    return PROMPT_TEMPLATE.format(
        glossary_block=glossary_block,
        context_block=context_block,
        words_per_second=WORDS_PER_SECOND,
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
        english = item["english"]
        if not isinstance(english, str) or not english.strip():
            raise ValueError(f"empty/invalid english for id {seg_id}")
        result[seg_id] = english.strip()

    if set(result.keys()) != set(expected_ids):
        raise ValueError(f"expected ids {expected_ids}, got {sorted(result.keys())}")
    return result


def translate_chunk(chunk, context_before, context_after, glossary, model, host, max_retries=3):
    expected_ids = [seg["id"] for seg in chunk]
    prompt = build_prompt(chunk, context_before, context_after, glossary)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_ollama(prompt, model, host)
            return validate_and_extract(raw, expected_ids)
        except Exception as e:
            last_error = repr(e)
            print(f"  retry {attempt}/{max_retries} for ids {expected_ids}: {last_error}")

    raise RuntimeError(f"translation failed for ids {expected_ids} after {max_retries} attempts: {last_error}")


def translate(episode_dir: Path, glossary_path: Path | None = None) -> list[dict]:
    segments_path = episode_dir / "segments.json"
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    config = yaml.safe_load(CONFIG_PATH.read_text())
    translation_cfg = config["translation"]
    ollama_cfg = config["models"]["ollama"]

    glossary = {}
    if glossary_path and glossary_path.exists():
        glossary = json.loads(glossary_path.read_text(encoding="utf-8"))

    chunk_size = translation_cfg["chunk_size"]
    context_n = translation_cfg["context_lines"]
    chunk_starts = list(range(0, len(segments), chunk_size))

    for n, start in enumerate(chunk_starts):
        chunk = segments[start:start + chunk_size]
        if all(seg.get("english") for seg in chunk):
            continue  # already translated -- resuming after a crash

        context_before = segments[max(0, start - context_n):start]
        context_after = segments[start + chunk_size:start + chunk_size + context_n]

        print(f"translating segments {chunk[0]['id']}-{chunk[-1]['id']} (chunk {n + 1}/{len(chunk_starts)})...")
        translations = translate_chunk(
            chunk, context_before, context_after, glossary,
            ollama_cfg["model"], ollama_cfg["host"],
        )
        for seg in chunk:
            seg["english"] = translations[seg["id"]]
        segments_path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")

    return segments


if __name__ == "__main__":
    episode_dir = Path(sys.argv[1])
    glossary_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = translate(episode_dir, glossary_path)
    print(f"translated {len(result)} segments")
