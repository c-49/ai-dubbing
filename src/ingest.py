"""Picks up new episodes dropped into each show's incoming/ folder, then runs
the full pipeline on every episode that isn't finished yet.

Usage: python src/ingest.py

Drop a video file into shows/<show-name>/incoming/<anything>.mp4 and run
this whenever you're ready to process it (and anything else waiting). The
video's filename (without extension) becomes the episode name:
  shows/vincenzo/incoming/ep5.mp4  ->  work/vincenzo/ep5/source.mp4

Also re-runs any episode under work/<show>/ that doesn't have a dubbed.mp4
yet, so this is also how you resume an interrupted overnight run.
"""
from pathlib import Path

import run_episode

ROOT = Path(__file__).resolve().parent.parent
SHOWS_DIR = ROOT / "shows"
WORK_DIR = ROOT / "work"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}


def ingest_new_videos() -> None:
    if not SHOWS_DIR.exists():
        return
    for show_dir in sorted(SHOWS_DIR.iterdir()):
        incoming = show_dir / "incoming"
        if not incoming.is_dir():
            continue
        for video in sorted(incoming.iterdir()):
            if video.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            episode_dir = WORK_DIR / show_dir.name / video.stem
            dest = episode_dir / "source.mp4"
            if dest.exists():
                print(f"skipping {video.name}: work/{show_dir.name}/{video.stem}/source.mp4 already exists")
                continue
            episode_dir.mkdir(parents=True, exist_ok=True)
            video.rename(dest)
            print(f"ingested {video.name} -> work/{show_dir.name}/{video.stem}/")


def find_pending_episodes() -> list[Path]:
    pending = []
    if not WORK_DIR.exists():
        return pending
    for show_dir in sorted(WORK_DIR.iterdir()):
        if not show_dir.is_dir():
            continue
        for episode_dir in sorted(show_dir.iterdir()):
            if (episode_dir / "source.mp4").exists() and not (episode_dir / "dubbed.mp4").exists():
                pending.append(episode_dir)
    return pending


def main() -> None:
    ingest_new_videos()
    pending = find_pending_episodes()
    if not pending:
        print("Nothing to do -- no new videos and no unfinished episodes.")
        return

    print(f"Processing {len(pending)} episode(s):")
    for p in pending:
        print(f"  {p.relative_to(ROOT)}")

    for episode_dir in pending:
        try:
            run_episode.run(episode_dir)
        except Exception as e:
            print(f"FAILED on {episode_dir} ({e!r}) -- continuing with the next episode")


if __name__ == "__main__":
    main()
