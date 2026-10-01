# Multi-Language Plan

Companion to `PLAN.md` (the dubbing pipeline) and `UI_PLAN.md` (the review
app). This file covers generalizing the pipeline beyond Korean-to-English:
Japanese and Korean as source languages, English and Spanish as target
languages.

## Goal
Support uploading a Japanese *or* Korean show and dubbing it into English
*or* Spanish, chosen per show. Today almost everything is hardcoded to
Korean-in, English-out -- this plan removes that so the language pair is a
per-show setting instead of an assumption baked into the code.

## What's actually hardcoded today (found by reading the code, not assumed)
- `transcribe.py`: `language="ko"` hardcoded on the Whisper call; segment
  fields are literally named `"korean"` / `"english"` in `segments.json`.
- `translate.py`: the LLM prompt is hardcoded ("Korean TV drama... into
  natural, spoken English"), and reads/writes those same `korean`/`english`
  keys.
- `timing.py`: has its *own* separate hardcoded Korean-to-English
  re-translation prompt, plus its own hardcoded `KPipeline(lang_code="a")`.
- `tts.py`: `KPipeline(lang_code="a")` hardcoded (American English).
- `review_ui/app.py`: the `VOICES` dict only lists English (American/British)
  Kokoro voices; `index.html`'s table headers literally say "Korean"/
  "English".
- `mix.py`: output track metadata labels say "English Dub"/"Original
  Korean".
- `shows/<show>/glossary.json`: flat `{korean-term: english-translation}`,
  implicitly tied to one target language.

## What's already in our favor
- faster-whisper large-v3 (already installed, already downloaded) natively
  transcribes Japanese and Spanish -- no new model needed, just stop
  hardcoding `language="ko"`.
- The installed Kokoro package already lists Spanish (`lang_code='e'`) as
  espeak-ng-backed, same mechanism as English -- espeak-ng is already
  installed (Milestone 0). Exact Spanish voice IDs (likely `ef_*`/`em_*`
  naming) still need confirming against Kokoro's HF model card at
  implementation time -- not treated as fact until verified.
- Diarization (pyannote) and mixing (ffmpeg levels/ducking) are language-
  agnostic already; no changes expected there beyond cosmetic metadata
  labels.

## Decision made
- **One target language per show, not several at once.** A show has a
  single fixed source+target language pair in its config. Re-dubbing to a
  different target language means re-running translation/TTS/mix for that
  show, not maintaining parallel dubs side by side. Simpler data model,
  matches the existing single-track design in `PLAN.md`.
- **Language list is config-driven, not hardcoded Python constants.** A
  small allowlist (source languages, target languages, and their Kokoro
  `lang_code`/voice-prefix mapping) lives in `config.yaml`, so adding a 5th
  language later is a config edit, not a code change across 6 files again.

## Milestones

### M7: Language configuration plumbing
- Add `source_language` / `target_language` fields to show creation (new
  dropdowns in the "New show" form), stored in a new
  `shows/<show>/config.json`. Validate against the config-driven allowlist.
- Backfill `shows/vincenzo/config.json` as `ko -> en` so the existing show
  keeps working unchanged.
- Rename segment fields from `korean`/`english` to generic `source_text`/
  `target_text` across `transcribe.py`, `translate.py`, `timing.py`,
  `tts.py`, `review_ui/app.py`, and `index.html` -- one clean rename now
  while only one real episode exists, instead of carrying two field-naming
  schemes forward indefinitely.
- **Done when:** creating a show asks for source/target language, and
  `work/vincenzo/test_clip` still loads correctly in the editor after the
  field rename.

### M8: Japanese transcription
- Parameterize `transcribe.py`'s `language=` from the show's
  `source_language` instead of hardcoding `"ko"`.
- When a Japanese test clip is available, run it through and tune VAD
  settings if Japanese pacing/pausing needs different silence thresholds
  than Korean did.
- **Done when:** Japanese transcription accuracy is judged against a test
  clip, same bar as the original Milestone 1 used for Korean.

### M9: Translation for ja/ko -> en/es
- Generalize `translate.py`'s prompt template to take full source/target
  language names instead of the hardcoded "Korean... English" wording.
- Make `WORDS_PER_SECOND` (used for length guidance) per-target-language --
  Spanish spoken pace differs from English and needs empirical tuning, not
  a guess.
- Apply the same generalization to `timing.py`'s separate re-translation
  prompt (it currently duplicates the Korean-to-English wording
  independently of `translate.py`).
- Evolve `glossary.json` from a flat mapping to
  `{"term": {"en": "...", "es": "..."}}` so one glossary can serve whichever
  target language is active, with a migration for the existing flat
  `shows/vincenzo/glossary.json`.
- **Risk to flag:** `qwen2.5:7b` was chosen specifically for its Korean
  handling (see `PLAN.md`/`README.md`). Japanese and Spanish output quality
  through the same model is unproven and may need a different Ollama model
  for one or both language pairs.
- **Done when:** translation quality is approved on test clips for each
  language pair actually in use.

### M10: Spanish TTS -- done
- Confirmed Kokoro's Spanish voice pack (`lang_code='e'`, espeak-ng-backed
  via `misaki.espeak`, unlike `'a'`'s bundled G2P) against the existing
  espeak-ng install with no new system dependency -- added as a permanent
  check in `check_setup.py`.
- Confirmed the real Spanish voice IDs against Kokoro-82M's HF repo file
  listing rather than guessing: only 3 exist today -- `ef_dora` (female),
  `em_alex`, `em_santa` (male) -- vs. 20 for English.
- Parameterized the hardcoded `KPipeline(lang_code="a")` call sites
  (`tts.py`, `timing.py`, and the review UI's own TTS pipeline, which also
  needed its single cached pipeline instance turned into one-per-language-
  code) off `target_language` via `config.yaml`'s `languages.target.<code>`.
- Added Spanish voice groups to the review UI's voice picker (`es_female`/
  `es_male`), scoped per show's own target language -- a Spanish-target
  show only ever sees Spanish voices.
- Moved `default_voice` from a single flat `models.tts.default_voice` to
  per-target-language (`languages.target.<code>.default_voice`).
- **Key realization that unblocked this milestone:** Spanish is only ever
  a *target* language in this project (dub output) -- there is no Spanish
  *source* clip, and none is needed. M10 dubs an existing Korean/Japanese
  test clip *into* Spanish, same as any other target language.
- **Done when:** a Spanish dub of a test clip plays with correctly assigned
  Spanish voices per speaker -- met by re-dubbing a copy of `vincenzo`'s
  existing Korean test clip to Spanish (`ko -> es`) end-to-end: translate,
  TTS (`ef_dora`/`em_alex`/`em_santa` per speaker), timing fit, and mix all
  completed with 0 segments flagged, and `mix.py`'s metadata correctly read
  "Spanish Dub"/"Original Korean".
- **Finding carried over from M9's name-consistency issue (Japanese):** the
  same problem hit Spanish on an empty glossary -- "빈센조 가사노" (Vincenzo
  Cassano) came out as "Sensoz" uncorrected. Fixed the same way: added the
  name to the show's glossary (`{"빈센조 가사노": {"es": "Vincenzo Cassano"}}`),
  re-translated, confirmed it now reads correctly. Not a code gap -- a
  reminder that a brand-new show's empty glossary is a real risk for any
  target language, not just Japanese.
- **New limitation found, not fixed:** this test clip has 4 speakers (2
  female, 2 male) but only 1 Spanish female voice exists -- two different
  female characters are forced to share `ef_dora`. Nothing to fix in code;
  it's a real constraint of Kokoro's current Spanish voice pack.

### M11: UI language-awareness
- Episode editor's table headers become dynamic ("Source (Japanese)" /
  "Target (Spanish)") instead of hardcoded "Korean"/"English".
- Show cards/pages get a language badge (e.g. "JA -> ES").
- `mix.py`'s output audio track metadata titles become dynamic instead of
  "English Dub"/"Original Korean".
- **Done when:** the whole UI reads correctly end-to-end for a non-Korean,
  non-English show.

### M12: Full non-Korean episode run
- Run one complete Japanese-or-Korean-source episode to Spanish end-to-end
  on a full episode, same bar as the original Milestone 7 (full-episode
  run) in `PLAN.md`.
- Confirm runtime estimates still roughly hold -- diarization/Demucs are
  language-agnostic so shouldn't shift, but transcription/translation/TTS
  timing may differ per language.

## Open items to revisit during implementation
- Per-target-language `words_per_second` value for Spanish (currently
  `2.8`, a placeholder) -- still needs empirical tuning against real
  Spanish TTS/timing output, the way English's `2.3` was tuned in the
  original Milestone 2. M10 validated the TTS *mechanics*, not this value.
- Whether `qwen2.5:7b` is adequate for Japanese and Spanish long-term: on
  the one Japanese and one Spanish test clip tried so far, translation was
  fluent and the pipeline completed cleanly, but both hit the same
  failure mode independently -- a character name garbled on an empty
  glossary (fixed by hand each time, not by the model). Not yet known
  whether that's the only failure mode or just the one small test clips
  happened to surface.
- Kokoro's Spanish voice pack has only 3 voices (1 female, 2 male) vs.
  English's 20 -- a show with more than 2 same-gender speakers will have
  to reuse a voice across different characters. No workaround planned;
  just a constraint to keep in mind when picking target languages for a
  many-speaker show.
