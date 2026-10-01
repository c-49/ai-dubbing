# Web UI Plan

Companion to `PLAN.md` (the dubbing pipeline itself). This file covers turning
`review_ui/` from a single-episode review page into a persistent, multi-show
app: drag-and-drop uploads, a shows/episodes browser, and a nicer episode view
with synced video playback.

## Goal
Replace the current workflow (drop a video into `shows/<show>/incoming/`, run
`ingest.py` from a terminal, launch `review_ui/app.py work/<show>/<episode>`
by hand) with one persistent local web app: `python review_ui/app.py`, no
arguments, browse everything from `http://127.0.0.1:5000`.

## Decisions made
- **Manual start, not auto-start.** Dropping a video onto a show just ingests
  it (copies the file, creates the work folder, probes metadata) and leaves
  it "Pending". You press "Start" (per episode, or "Process all pending") to
  queue it for the pipeline. Keeps you in control of when a 1.5-2hr run kicks
  off.
- **Side-by-side video players.** Source video and dubbed video shown as two
  separate players on the episode page, each independently seekable, but a
  timestamp click seeks/plays both together.
- **Full replacement, not backwards compatible.** The old
  `review_ui/app.py work/<show>/<episode>` single-episode launch style goes
  away entirely in favor of the persistent multi-show server.
- **Single background worker.** Only one episode processes at a time, in
  either the UI queue or `ingest.py`'s batch run, per the 16GB RAM constraint
  in `PLAN.md` (never load Whisper and the LLM at once). `ingest.py` keeps
  working for headless/overnight CLI use and shares the same queue/state so
  the UI and CLI don't double-process an episode.

## Milestones

### M1: Multi-show server backbone
- Turn `review_ui/app.py` into a persistent server with no required CLI arg.
- Routes: `/` (shows list), `/show/<name>` (episode list),
  `/show/<name>/<episode>` (the existing segment editor, moved here as-is).
- No new features yet -- pure restructure. Confirms the new routing/data
  layer works before anything else is built on top of it.
- **Done when:** the server starts with just `python review_ui/app.py`, and
  the existing segment-editing/voice/remix functionality for `test_clip`
  works unchanged at its new URL.

### M2: Shows UI
- Home page lists shows as cards (from `shows/*` dirs).
- "New show" button/form creates `shows/<name>/` with an empty
  `glossary.json`.
- Each show card links to its episode list.
- **Done when:** you can create a show from the browser and see it listed,
  with no manual folder creation.

### M3: Drag-and-drop upload + manual-start queue
- Drag a video onto a show's drop zone -> uploads via the browser (multipart
  POST, with an upload progress indicator) to
  `work/<show>/<episode-name>/source.mp4`. Episode name derived from the
  filename, same as `ingest.py` does today.
- Probe the video on upload (ffprobe) for duration, resolution -- store as
  episode metadata for display.
- New episode appears with status "Pending" -- nothing runs automatically.
- "Start" button per pending episode, plus a "Process all pending" button.
- A single background worker thread/process consumes a queue one episode at
  a time and runs `run_episode.run()` on each.
- `ingest.py` keeps working from the terminal for batch/overnight use,
  reading and writing the same queue/status state as the UI so the two don't
  collide on the same episode.
- **Done when:** dropping a video onto a show in the browser, then clicking
  Start, produces the same result as the current `incoming/` + `ingest.py`
  flow -- without touching a terminal.

### M4: Status & progress
- Episode card/page shows: Pending -> Queued -> Running (stage N/8, which
  stage) -> Done / Failed.
- Auto-refresh while something is running (polling is fine).
- Log tail visible on the episode page while it runs or after it fails.
- **Done when:** you can drop a video, hit Start, and watch it progress
  through all 8 stages from the browser alone, including seeing a clear
  error state if a stage fails.

### M5: Video playback + timestamp UX
- Episode page gets two video players side by side: source (`source.mp4`)
  and dub (`dubbed.mp4`), only once `dubbed.mp4` exists.
- All timestamps in the segment table rendered as `hh:mm:ss` instead of raw
  seconds.
- Clicking a segment's timestamp seeks both players to that time and plays
  them (muting/pausing handled sensibly so they don't both blast audio at
  once -- likely: clicking a timestamp plays the dub by default, with a
  per-row or global toggle for "preview original instead").
- **Done when:** you can click any segment's timestamp and land on that
  moment in both the source and dub videos.

### M6: Polish
- Delete an episode (with confirmation) / re-ingest a replacement file.
- Retry a failed episode from where it left off (stages already support
  resuming -- just needs a UI button that calls `run_episode.run()` again).
- Make the source/dub video players sticky (`position: sticky`) so they stay
  on screen while scrolling through the segment table -- raised after M5
  shipped, not yet done.
- Anything else that falls out of actually using M1-M5 day to day.

## Open items to revisit during implementation
- Exact concurrent-access story between `ingest.py` (CLI) and the UI worker
  if both are run at the same time -- likely a simple lock file in `work/`
  naming the episode currently processing, checked by both.
- Where upload temp files land during a large multipart upload (write
  directly to `work/<show>/<episode>/source.mp4` via streaming, not buffered
  in memory, since episodes can be 30-60 min of video).
- **Manual timestamp editing -- implemented.** Surfaced while reviewing a
  real multi-speaker dub where diarization/VAD occasionally mis-times a
  segment (see LANGUAGE_PLAN.md M12 notes on the Japanese test clip). The
  Time column now has small start/end number inputs (step 0.1s) alongside
  the existing seek/play button. Editing re-synthesizes and re-fits the
  edited segment (and the previous segment too, since its available slot
  depends on this segment's start) from a fresh, not-yet-sped-up clip before
  calling `fit_segment` again -- `atempo` isn't idempotent, so always
  resetting first avoids compounding speedups.
  - **Known limitation:** editing a segment's *end* can change whether the
    *next* segment's source lines "originally overlapped" (which controls
    whether timing is allowed to shorten it) -- the next segment isn't
    automatically re-fit when that happens, so it can go stale until it's
    next edited or the episode is fully re-run. Only the immediate
    start/previous-segment interaction is kept in sync automatically.
  - As with text/voice edits, a changed timestamp doesn't automatically
    regenerate `dubbed.mp4` -- hit "Re-run mix" afterward.
