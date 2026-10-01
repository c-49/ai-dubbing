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

VOICES = {
    "us_female": ["af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
                  "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"],
    "us_male": ["am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
                "am_michael", "am_onyx", "am_puck", "am_santa"],
    "uk_female": ["bf_alice", "bf_emma", "bf_isabella", "bf_lily"],
    "uk_male": ["bm_daniel", "bm_fable", "bm_george", "bm_lewis"],
}

app = Flask(__name__)
_tts_pipeline = None

# Single background worker: only one episode's pipeline runs at a time (also
# enforced across processes by run_episode.py's work/.pipeline.lock, so a
# manually-run ingest.py won't collide with this either).
_work_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_queued_keys: set[tuple[str, str]] = set()
_queue_lock = threading.Lock()


def enqueue_episode(show: str, episode: str) -> bool:
    key = (show, episode)
    with _queue_lock:
        if key in _queued_keys:
            return False
        _queued_keys.add(key)

    episode_dir = WORK_DIR / show / episode
    status = load_json(episode_dir / "status.json") or {}
    status["status"] = "queued"
    status["error"] = None
    save_json(episode_dir / "status.json", status)
    _work_queue.put(key)
    return True


def _worker_loop() -> None:
    while True:
        show, episode = _work_queue.get()
        with _queue_lock:
            _queued_keys.discard((show, episode))
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


def get_tts_pipeline() -> KPipeline:
    global _tts_pipeline
    if _tts_pipeline is None:
        _tts_pipeline = KPipeline(lang_code="a")
    return _tts_pipeline


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


def synthesize_clip(episode_dir: Path, seg: dict) -> None:
    pipeline = get_tts_pipeline()
    chunks = [audio for _, _, audio in pipeline(seg["english"], voice=seg["voice"])]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    out_path = episode_dir / seg["audio_path"]
    sf.write(out_path, audio, TTS_SAMPLE_RATE)


def refit_segment(episode_dir: Path, segments: list[dict], idx: int) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    ollama_cfg = config["models"]["ollama"]
    fit_segment(
        episode_dir, segments, idx, config["timing"]["max_speedup"],
        ollama_cfg["model"], ollama_cfg["host"], get_tts_pipeline(),
    )


@app.get("/")
def home():
    shows = [{"name": name, "episode_count": len(list_episodes(name))} for name in list_shows()]
    return render_template("home.html", shows=shows)


@app.post("/api/shows")
def create_show():
    name = (request.get_json(silent=True) or {}).get("name", "").strip()
    if not SHOW_NAME_RE.match(name):
        return jsonify({"error": "show name must start with a letter/number and contain only "
                                  "letters, numbers, '-' or '_'"}), 400
    if name in list_shows():
        return jsonify({"error": f"show '{name}' already exists"}), 400

    show_dir = SHOWS_DIR / name
    show_dir.mkdir(parents=True)
    save_json(show_dir / "glossary.json", {})

    return jsonify({"name": name})


@app.get("/show/<show>")
def show_page(show):
    if show not in list_shows():
        abort(404, description=f"no show '{show}'")
    return render_template("show.html", show=show)


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
        if existing_status in ("queued", "running"):
            return jsonify({"error": f"cannot replace -- episode is currently {existing_status}"}), 400
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
    if status in ("queued", "running"):
        return jsonify({"error": f"cannot delete -- episode is currently {status}"}), 400
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
    return jsonify({
        "episode": episode,
        "show": show,
        "segments": segments,
        "speakers": speakers,
        "voices": VOICES,
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
    if "english" in data and data["english"] != seg["english"]:
        seg["english"] = data["english"]
        regenerate = True
    if "voice" in data and data["voice"] != seg["voice"]:
        seg["voice"] = data["voice"]
        regenerate = True
    if "needs_review" in data:
        seg["needs_review"] = bool(data["needs_review"])

    if regenerate:
        synthesize_clip(episode_dir, seg)
        refit_segment(episode_dir, segments, idx)

    save_json(episode_dir / "segments.json", segments)
    return jsonify(segments[idx])


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

    segments = load_json(episode_dir / "segments.json")
    for idx, seg in enumerate(segments):
        if seg.get("speaker") == speaker and seg.get("english"):
            seg["voice"] = voice
            synthesize_clip(episode_dir, seg)
            refit_segment(episode_dir, segments, idx)
    save_json(episode_dir / "segments.json", segments)

    return jsonify({"speaker": speaker, "voice": voice, "segments": segments})


@app.post("/api/<show>/<episode>/remix")
def remix(show, episode):
    episode_dir = get_episode_dir(show, episode)
    out_path = run_mix(episode_dir)
    return jsonify({"ok": True, "path": str(out_path)})


@app.get("/audio/<show>/<episode>/<path:relpath>")
def serve_audio(show, episode, relpath):
    episode_dir = get_episode_dir(show, episode)
    return send_from_directory(episode_dir, relpath)


if __name__ == "__main__":
    _reconcile_stale_statuses()
    threading.Thread(target=_worker_loop, daemon=True).start()
    app.run(debug=True, port=5000, use_reloader=False)
