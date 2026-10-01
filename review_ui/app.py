"""Local web app for browsing shows/episodes and reviewing dubs.

Usage: python review_ui/app.py
Open http://127.0.0.1:5000 in a browser.

Home page lists shows (shows/<name>/), each show page lists its episodes
(work/<show>/<episode>/), and each episode page is the review editor: read/
edit segment text, pick a voice per speaker (with sample clips to listen
to), play back individual TTS clips, filter to flagged lines, and re-run
the final mix -- without touching the terminal.
"""
import queue
import re
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory, render_template
from werkzeug.utils import secure_filename

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import pathfix  # noqa: E402,F401
import json  # noqa: E402
import yaml  # noqa: E402
from kokoro import KPipeline  # noqa: E402
import soundfile as sf  # noqa: E402
import numpy as np  # noqa: E402

import ingest  # noqa: E402 (just file-extension constants + ingest helpers, lightweight)
from timing import fit_segment, clip_duration  # noqa: E402
from mix import mix as run_mix  # noqa: E402

ROOT_DIR = SRC_DIR.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"
SHOWS_DIR = ROOT_DIR / "shows"
WORK_DIR = ROOT_DIR / "work"
TTS_SAMPLE_RATE = 24000

# Work directories that aren't shows (scratch space used by other scripts).
IGNORED_SHOW_DIRS = {"_setup_check"}

# Keep show names filesystem- and URL-safe (used directly as a folder name
# under both shows/ and work/).
SHOW_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")

# Keyed by target_language, then voice group -- so a show's voice picker only
# ever offers voices for its own target language (see get_data()). Spanish
# only has 3 voices total (vs. English's 20), confirmed against Kokoro-82M's
# HF repo file listing, not guessed -- see LANGUAGE_PLAN.md M10.
VOICES = {
    "en": {
        "us_female": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                      "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
        "us_male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                    "am_michael", "am_onyx", "am_puck", "am_santa"],
        "uk_female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
        "uk_male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
    },
    "es": {
        "es_female": ["ef_dora"],
        "es_male": ["em_alex", "em_santa"],
    },
}

app = Flask(__name__)
_tts_pipelines: dict[str, "KPipeline"] = {}  # keyed by Kokoro lang_code ("a", "e", ...)

# Single background worker: only one episode's pipeline runs at a time (also
# enforced across processes by run_episode.py's work/.pipeline.lock, so a
# manually-run ingest.py won't collide with this either).
_work_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_queued_keys: set[tuple[str, str]] = set()
_cancelled_keys: set[tuple[str, str]] = set()
_queue_lock = threading.Lock()


def enqueue_episode(show: str, episode: str) -> bool:
    key = (show, episode)
    with _queue_lock:
        if key in _queued_keys:
            return False
        _queued_keys.add(key)
        _cancelled_keys.discard(key)

    episode_dir = WORK_DIR / show / episode
    status = load_json(episode_dir / "status.json") or {}
    status["status"] = "queued"
    status["error"] = None
    save_json(episode_dir / "status.json", status)
    _work_queue.put(key)
    return True


def cancel_queued_episode(show: str, episode: str) -> bool:
    """Lets a still-waiting-in-line episode be deleted/replaced without
    waiting for its turn. Can't un-queue from queue.Queue directly, so the
    worker just skips it when its turn comes (see _worker_loop)."""
    key = (show, episode)
    with _queue_lock:
        if key not in _queued_keys:
            return False
        _queued_keys.discard(key)
        _cancelled_keys.add(key)
    return True


def _worker_loop() -> None:
    while True:
        key = _work_queue.get()
        show, episode = key
        with _queue_lock:
            _queued_keys.discard(key)
            cancelled = key in _cancelled_keys
            _cancelled_keys.discard(key)
        if cancelled:
            print(f"[worker] skipping cancelled {show}/{episode}", flush=True)
            _work_queue.task_done()
            continue
        episode_dir = WORK_DIR / show / episode
        print(f"[worker] starting {show}/{episode}", flush=True)
        try:
            # Imported lazily (not at module load) -- it pulls in torch,
            # demucs, pyannote and faster-whisper, which we don't want to pay
            # for just to browse shows/episodes.
            import run_episode
            run_episode.run(episode_dir)
        except Exception as e:
            print(f"[worker] {show}/{episode} failed: {e!r}", flush=True)
        finally:
            _work_queue.task_done()


def _reconcile_stale_statuses() -> None:
    """On server startup, reset any episode left 'queued'/'running' by a
    previous process (crash, restart) back to 'pending' -- it isn't actually
    in this process's in-memory queue, so it would otherwise be stuck."""
    for show in list_shows():
        for episode in list_episodes(show):
            status_path = WORK_DIR / show / episode / "status.json"
            status = load_json(status_path)
            if status and status.get("status") in ("queued", "running"):
                status["status"] = "pending"
                save_json(status_path, status)


def get_tts_pipeline(kokoro_lang_code: str = "a") -> KPipeline:
    if kokoro_lang_code not in _tts_pipelines:
        _tts_pipelines[kokoro_lang_code] = KPipeline(lang_code=kokoro_lang_code)
    return _tts_pipelines[kokoro_lang_code]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_shows() -> list[str]:
    names = set()
    for base in (SHOWS_DIR, WORK_DIR):
        if base.exists():
            names.update(p.name for p in base.iterdir() if p.is_dir())
    names -= IGNORED_SHOW_DIRS
    return sorted(names)


def list_episodes(show: str) -> list[str]:
    show_dir = WORK_DIR / show
    if not show_dir.exists():
        return []
    return sorted(p.name for p in show_dir.iterdir() if p.is_dir())


def get_episode_dir(show: str, episode: str) -> Path:
    episode_dir = WORK_DIR / show / episode
    if not episode_dir.is_dir():
        abort(404, description=f"no episode work/{show}/{episode}")
    return episode_dir


def get_episode_status(episode_dir: Path) -> dict:
    """Pending/queued/running/done/failed, plus metadata/error if known.
    Falls back to inferring from the files on disk for episodes that predate
    status.json (or were ingested by hand)."""
    status = load_json(episode_dir / "status.json") or {}
    if "status" not in status:
        if (episode_dir / "dubbed.mp4").exists():
            status["status"] = "done"
        elif (episode_dir / "source.mp4").exists():
            status["status"] = "pending"
        else:
            status["status"] = "unknown"
    return status


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(result.stdout)
    duration = float(data.get("format", {}).get("duration") or 0) or None
    width = height = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width, height = stream.get("width"), stream.get("height")
            break
    return {"duration_sec": round(duration, 1) if duration else None, "width": width, "height": height}


def get_voices_path(show: str) -> Path:
    return SHOWS_DIR / show / "voices.json"


def get_language_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())["languages"]


def get_show_config(show: str) -> dict | None:
    return load_json(SHOWS_DIR / show / "config.json")


def get_show_language_info(show: str) -> dict:
    """Resolves a show's source/target language codes (defaulting to ko/en
    for shows predating config.json) plus their display names and the
    target language's Kokoro settings, for the UI's language badges/headers,
    TTS synthesis, and passing into mix.py."""
    show_config = get_show_config(show) or {}
    source_language = show_config.get("source_language", "ko")
    target_language = show_config.get("target_language", "en")
    languages = get_language_config()
    target_cfg = languages["target"][target_language]
    return {
        "source_language": source_language,
        "target_language": target_language,
        "source_language_name": languages["source"][source_language]["name"],
        "target_language_name": target_cfg["name"],
        "kokoro_lang_code": target_cfg["kokoro_lang_code"],
        "default_voice": target_cfg["default_voice"],
        "words_per_second": target_cfg["words_per_second"],
    }


def synthesize_clip(episode_dir: Path, seg: dict, kokoro_lang_code: str = "a") -> None:
    pipeline = get_tts_pipeline(kokoro_lang_code)
    chunks = [audio for _, _, audio in pipeline(seg["target_text"], voice=seg["voice"])]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    out_path = episode_dir / seg["audio_path"]
    sf.write(out_path, audio, TTS_SAMPLE_RATE)


def refit_segment(show: str, episode_dir: Path, segments: list[dict], idx: int) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    ollama_cfg = config["models"]["ollama"]

    lang_info = get_show_language_info(show)

    fit_segment(
        episode_dir, segments, idx, config["timing"]["max_speedup"],
        ollama_cfg["model"], ollama_cfg["host"], get_tts_pipeline(lang_info["kokoro_lang_code"]),
        lang_info["source_language_name"], lang_info["target_language_name"], lang_info["words_per_second"],
        # Never auto-rewrite text behind the reviewer's back -- every call
        # here follows an interactive edit (text, voice, or timestamp), so a
        # human has either just chosen this wording on purpose or is relying
        # on it staying put. If it still doesn't fit, flag it and leave it
        # for them to resolve (e.g. nudge the timestamp, or accept the
        # overlap) rather than silently rewriting it again.
        allow_auto_shorten=False,
    )


@app.get("/")
def home():
    shows = [
        {"name": name, "episode_count": len(list_episodes(name)), **get_show_language_info(name)}
        for name in list_shows()
    ]
    return render_template("home.html", shows=shows, languages=get_language_config())


@app.get("/api/languages")
def list_languages():
    return jsonify(get_language_config())


@app.post("/api/shows")
def create_show():
    body = request.get_json(silent=True) or {}
    name = body.get("name", "").strip()
    source_language = body.get("source_language", "").strip()
    target_language = body.get("target_language", "").strip()

    if not SHOW_NAME_RE.match(name):
        return jsonify({"error": "show name must start with a letter/number and contain only "
                                  "letters, numbers, '-' or '_'"}), 400
    if name in list_shows():
        return jsonify({"error": f"show '{name}' already exists"}), 400

    languages = get_language_config()
    if source_language not in languages["source"]:
        return jsonify({"error": f"source language must be one of {sorted(languages['source'])}"}), 400
    if target_language not in languages["target"]:
        return jsonify({"error": f"target language must be one of {sorted(languages['target'])}"}), 400

    show_dir = SHOWS_DIR / name
    show_dir.mkdir(parents=True)
    save_json(show_dir / "glossary.json", {})
    save_json(show_dir / "config.json", {
        "source_language": source_language,
        "target_language": target_language,
    })

    return jsonify({"name": name, "source_language": source_language, "target_language": target_language})


@app.get("/show/<show>")
def show_page(show):
    if show not in list_shows():
        abort(404, description=f"no show '{show}'")
    return render_template("show.html", show=show, **get_show_language_info(show))


@app.get("/api/<show>/episodes")
def list_episodes_api(show):
    if show not in list_shows():
        abort(404, description=f"no show '{show}'")
    episodes = [
        {"name": name, **get_episode_status(WORK_DIR / show / name)}
        for name in list_episodes(show)
    ]
    return jsonify({"show": show, "episodes": episodes})


@app.post("/api/<show>/upload")
def upload_episode(show):
    if show not in list_shows():
        abort(404, description=f"no show '{show}'")

    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "no file uploaded"}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in ingest.VIDEO_EXTENSIONS:
        return jsonify({"error": f"unsupported file type '{suffix}' -- expected one of "
                                  f"{sorted(ingest.VIDEO_EXTENSIONS)}"}), 400

    episode = Path(filename).stem
    if not episode:
        return jsonify({"error": "could not derive an episode name from that filename"}), 400

    episode_dir = WORK_DIR / show / episode
    dest = episode_dir / "source.mp4"
    if dest.exists():
        if request.form.get("replace") != "1":
            return jsonify({"error": f"episode '{episode}' already exists for this show",
                             "exists": True}), 409
        existing_status = get_episode_status(episode_dir)["status"]
        if existing_status == "running":
            return jsonify({"error": "cannot replace -- episode is currently running"}), 400
        if existing_status == "queued":
            cancel_queued_episode(show, episode)  # still waiting in line -- safe to pull out
        shutil.rmtree(episode_dir)  # discard all prior stage output, not just source.mp4

    episode_dir.mkdir(parents=True, exist_ok=True)
    file.save(dest)  # streamed to disk by Werkzeug, not buffered fully in memory

    try:
        metadata = probe_video(dest)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        metadata = {"duration_sec": None, "width": None, "height": None}

    save_json(episode_dir / "status.json", {
        "status": "pending",
        "error": None,
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
    })

    return jsonify({"episode": episode, **metadata})


@app.delete("/api/<show>/<episode>")
def delete_episode(show, episode):
    episode_dir = get_episode_dir(show, episode)
    status = get_episode_status(episode_dir)["status"]
    if status == "running":
        return jsonify({"error": "cannot delete -- episode is currently running"}), 400
    if status == "queued":
        cancel_queued_episode(show, episode)  # still waiting in line -- safe to pull out
    shutil.rmtree(episode_dir)
    return jsonify({"ok": True})


@app.post("/api/<show>/<episode>/start")
def start_episode(show, episode):
    episode_dir = get_episode_dir(show, episode)
    status = get_episode_status(episode_dir)["status"]
    if status not in ("pending", "failed"):
        return jsonify({"error": f"episode is already {status}"}), 400
    enqueue_episode(show, episode)
    return jsonify({"status": "queued"})


@app.post("/api/<show>/process-all")
def process_all(show):
    if show not in list_shows():
        abort(404, description=f"no show '{show}'")
    queued = []
    for episode in list_episodes(show):
        status = get_episode_status(WORK_DIR / show / episode)["status"]
        if status in ("pending", "failed") and enqueue_episode(show, episode):
            queued.append(episode)
    return jsonify({"queued": queued})


@app.get("/show/<show>/<episode>")
def episode_page(show, episode):
    get_episode_dir(show, episode)
    return render_template("index.html", show=show, episode=episode)


@app.get("/api/<show>/<episode>/status")
def episode_status_api(show, episode):
    episode_dir = get_episode_dir(show, episode)
    return jsonify({"name": episode, **get_episode_status(episode_dir)})


@app.get("/api/<show>/<episode>/log")
def episode_log(show, episode):
    episode_dir = get_episode_dir(show, episode)
    log_path = episode_dir / "run.log"
    if not log_path.exists():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[-200:]), 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/api/<show>/<episode>/data")
def get_data(show, episode):
    episode_dir = get_episode_dir(show, episode)
    segments = load_json(episode_dir / "segments.json") or []
    speakers = load_json(episode_dir / "speakers.json") or {}
    lang_info = get_show_language_info(show)
    return jsonify({
        "episode": episode,
        "show": show,
        "segments": segments,
        "speakers": speakers,
        # Scoped to this show's own target language -- a Spanish-target show
        # should only ever be offered Spanish voices, not a mixed list.
        "voices": VOICES[lang_info["target_language"]],
        **lang_info,
    })


@app.post("/api/<show>/<episode>/segment/<int:seg_id>")
def update_segment(show, episode, seg_id):
    episode_dir = get_episode_dir(show, episode)
    data = request.get_json()
    segments = load_json(episode_dir / "segments.json")
    idx = next((i for i, s in enumerate(segments) if s["id"] == seg_id), None)
    if idx is None:
        return jsonify({"error": "segment not found"}), 404
    seg = segments[idx]

    regenerate = False
    if "target_text" in data and data["target_text"] != seg["target_text"]:
        seg["target_text"] = data["target_text"]
        regenerate = True
    if "voice" in data and data["voice"] != seg["voice"]:
        seg["voice"] = data["voice"]
        regenerate = True
    if "needs_review" in data:
        seg["needs_review"] = bool(data["needs_review"])

    lang_info = get_show_language_info(show)
    kokoro_lang_code = lang_info["kokoro_lang_code"]

    if "ignored" in data:
        new_ignored = bool(data["ignored"])
        if new_ignored != bool(seg.get("ignored", False)):
            seg["ignored"] = new_ignored
            # Un-ignoring a line that was flagged before TTS ever ran on it
            # (so it never got a clip) needs one now, or it'd silently stay
            # mute -- give it the conventional path and a voice so the usual
            # synthesize+refit below picks it up. A line that already has a
            # clip (ignored after the fact) just needs no further action:
            # mix.py already skips "ignored" clips regardless of audio_path.
            if not new_ignored and not seg.get("audio_path") and seg.get("target_text"):
                seg["audio_path"] = f"tts/{seg['id']:04d}.wav"
                seg["voice"] = seg.get("voice") or lang_info["default_voice"]
                regenerate = True

    retimed = False
    if "start" in data or "end" in data:
        new_start = float(data.get("start", seg["start"]))
        new_end = float(data.get("end", seg["end"]))
        if new_start < 0 or new_end <= new_start:
            return jsonify({"error": "start must be 0 or greater and less than end"}), 400
        if (new_start, new_end) != (seg["start"], seg["end"]):
            seg["start"], seg["end"] = new_start, new_end
            retimed = True

    affected = [seg]
    if (regenerate or retimed) and seg.get("audio_path") and not seg.get("ignored"):
        synthesize_clip(episode_dir, seg, kokoro_lang_code)
        refit_segment(show, episode_dir, segments, idx)

    # Moving this segment's start also changes the previous segment's
    # available slot (its dub clip must fit before THIS segment's new
    # start) -- refit it too, from a freshly-synthesized (not-yet-sped-up)
    # copy of its clip, since atempo isn't idempotent (fit_segment's own
    # docstring: mutates the clip on disk at most once per call).
    if retimed and idx > 0 and segments[idx - 1].get("audio_path") and not segments[idx - 1].get("ignored"):
        prev = segments[idx - 1]
        synthesize_clip(episode_dir, prev, kokoro_lang_code)
        refit_segment(show, episode_dir, segments, idx - 1)
        affected.append(prev)

    save_json(episode_dir / "segments.json", segments)
    return jsonify({"segments": affected})


@app.post("/api/<show>/<episode>/speaker/<speaker>/voice")
def update_speaker_voice(show, episode, speaker):
    episode_dir = get_episode_dir(show, episode)
    voice = request.get_json()["voice"]

    speakers = load_json(episode_dir / "speakers.json") or {}
    if speaker in speakers:
        speakers[speaker]["voice"] = voice
        save_json(episode_dir / "speakers.json", speakers)

    voices_path = get_voices_path(show)
    voices = load_json(voices_path) or {}
    voices[speaker] = voice
    voices_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(voices_path, voices)

    kokoro_lang_code = get_show_language_info(show)["kokoro_lang_code"]

    segments = load_json(episode_dir / "segments.json")
    for idx, seg in enumerate(segments):
        if seg.get("speaker") == speaker and seg.get("target_text"):
            seg["voice"] = voice  # recorded either way, so it's already right if un-ignored later
            if not seg.get("ignored"):
                synthesize_clip(episode_dir, seg, kokoro_lang_code)
                refit_segment(show, episode_dir, segments, idx)
    save_json(episode_dir / "segments.json", segments)

    return jsonify({"speaker": speaker, "voice": voice, "segments": segments})


@app.post("/api/<show>/<episode>/remix")
def remix(show, episode):
    episode_dir = get_episode_dir(show, episode)
    lang_info = get_show_language_info(show)
    out_path = run_mix(episode_dir, lang_info["source_language"], lang_info["target_language"])
    return jsonify({"ok": True, "path": str(out_path)})


@app.get("/audio/<show>/<episode>/<path:relpath>")
def serve_audio(show, episode, relpath):
    episode_dir = get_episode_dir(show, episode)
    return send_from_directory(episode_dir, relpath)


if __name__ == "__main__":
    _reconcile_stale_statuses()
    threading.Thread(target=_worker_loop, daemon=True).start()
    app.run(debug=True, port=5000, use_reloader=False)
