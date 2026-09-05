# Anki French Audio Generator

Generates French pronunciation audio for an Anki deck-import CSV using the
ElevenLabs text-to-speech API, and writes the `.mp3` files directly into your
Anki media folder.

## Input format

A headerless CSV, one card per row:

```
bed,un lit
fox,le renard
```

- **Column 1** — English word (front of the card).
- **Column 2** — French word (back of the card).

A row is treated as **already having audio** if column 2 contains an Anki sound
tag, e.g. `le chat [sound:le_chat.mp3]`. Such rows are left untouched.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The virtualenv is
created automatically on first run; to create it explicitly:

```sh
uv sync
```

Configuration lives in `.env` (already populated):

| Variable | Meaning |
| --- | --- |
| `ELEVENLABS_API_KEY` | Your ElevenLabs API key. |
| `ELEVENLABS_VOICE_ID` | French voice used for synthesis. |
| `ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` (required for French). |
| `ANKI_MEDIA_DIR` | Absolute path to the profile's `collection.media` folder. |

## Usage

```sh
uv run python src/generate_audio.py path/to/deck.csv
```

This writes:

- generated `.mp3` files into `ANKI_MEDIA_DIR`, and
- `[sound:...]` tags appended to the input CSV **in place**.

Then **re-import the CSV into Anki** to attach the audio to the cards.

Options:

- `-j, --workers N` — parallel workers (default: 4). Keep this at or below your
  ElevenLabs plan's concurrent-request limit (free/starter plans allow 5).
  Requests that still hit the concurrency cap (HTTP 429) are retried
  automatically with exponential backoff.

Reruns are safe: rows that already have a sound tag are skipped, so you can
re-run on a partially-processed file to fill in only what failed before.

## Choosing a French voice

```sh
uv run python src/list_voices.py
```

Copy a `voice_id` from the output into `ELEVENLABS_VOICE_ID` in `.env`.
