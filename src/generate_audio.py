"""Generate French audio for an Anki deck-import CSV via the ElevenLabs API.

Input CSV: each row is `<english_front>,<french_back>`.
The French cell may already contain an Anki sound tag, e.g. `un lit [sound:un_lit.mp3]`.

For every row that does NOT already have a `[sound:...]` tag, this script:
  1. synthesizes the French text with a French ElevenLabs voice,
  2. writes `<slug>.mp3` into the Anki media folder,
  3. appends ` [sound:<slug>.mp3]` to the French cell.

Input CSVs are discovered from git: every `.csv` file that is modified
(staged or unstaged) or untracked relative to HEAD is processed.

Rows are processed in parallel (network-bound work). Each input CSV is updated
in place, so you re-import the same file into Anki. The .mp3 files land directly
in the media folder, so Anki picks them up on import.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError
from tqdm import tqdm

SOUND_TAG_RE = re.compile(r"\[sound:[^\]]+\]")

# ElevenLabs concurrency-limit handling: retry 429s with exponential backoff.
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds

# Default parallelism kept at/under the common free/starter concurrency cap (5).
DEFAULT_WORKERS = 4


def slugify(text: str) -> str:
    """Turn French text into a filesystem-safe, ASCII snake_case filename stem.

    Matches the existing convention in the media folder (e.g. "un lit" -> "un_lit").
    """
    # Strip accents: "à" -> "a", "é" -> "e", etc.
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    # Collapse any run of non-alphanumeric characters into a single underscore.
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    return slug or "audio"


def has_audio(french_cell: str) -> bool:
    """True if the French cell already references an Anki sound file."""
    return bool(SOUND_TAG_RE.search(french_cell))


def french_text_only(french_cell: str) -> str:
    """The spoken text: the cell with any existing sound tag(s) removed."""
    return SOUND_TAG_RE.sub("", french_cell).strip()


def unique_media_path(media_dir: Path, stem: str, lock: threading.Lock) -> Path:
    """Reserve a non-colliding `<stem>.mp3` path in the media folder.

    Guarded by a lock so parallel workers don't hand out the same name. We create
    an empty placeholder file to claim the name before releasing the lock.
    """
    with lock:
        candidate = media_dir / f"{stem}.mp3"
        n = 2
        while candidate.exists():
            candidate = media_dir / f"{stem}_{n}.mp3"
            n += 1
        candidate.touch()  # claim the name
        return candidate


class AudioGenerator:
    def __init__(
        self,
        client: ElevenLabs,
        media_dir: Path,
        voice_id: str,
        model_id: str,
    ) -> None:
        self.client = client
        self.media_dir = media_dir
        self.voice_id = voice_id
        self.model_id = model_id
        self._name_lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        # ElevenLabs enforces a per-subscription concurrency cap (e.g. 5 parallel
        # requests). If we still hit it, back off and retry rather than failing
        # the row outright.
        for attempt in range(MAX_RETRIES):
            try:
                stream = self.client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    text=text,
                    language_code="fr",
                    output_format="mp3_44100_128",
                )
                return b"".join(stream)
            except ApiError as exc:
                if exc.status_code != 429 or attempt == MAX_RETRIES - 1:
                    raise
                # Exponential backoff with jitter derived from the text (no RNG,
                # keeps retries staggered across concurrent workers).
                delay = RETRY_BASE_DELAY * (2**attempt) + (len(text) % 5) * 0.1
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover

    def process_row(self, index: int, english: str, french_cell: str) -> tuple[int, str, str | None]:
        """Return (index, updated_french_cell, error). Error is None on success/skip."""
        if has_audio(french_cell):
            return index, french_cell, None

        text = french_text_only(french_cell)
        if not text:
            return index, french_cell, "empty French text"

        path = unique_media_path(self.media_dir, slugify(text), self._name_lock)
        try:
            audio = self.synthesize(text)
            if not audio:
                raise RuntimeError("ElevenLabs returned empty audio")
            path.write_bytes(audio)
        except Exception as exc:  # noqa: BLE001 - report per row, don't abort the batch
            # Remove the placeholder we claimed so a rerun can reuse the name.
            path.unlink(missing_ok=True)
            return index, french_cell, str(exc)

        updated = f"{french_cell.rstrip()} [sound:{path.name}]"
        return index, updated, None


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.reader(f) if any(cell.strip() for cell in row)]


def write_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def git(*args: str, cwd: Path | None = None) -> list[str]:
    """Run a git command and return its non-empty output lines."""
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except FileNotFoundError:
        sys.exit("error: git is not installed or not on PATH")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: git {' '.join(args)} failed: {exc.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def changed_csv_files() -> list[Path]:
    """CSV files that are modified (staged or unstaged) or untracked vs. HEAD."""
    repo_root = Path(git("rev-parse", "--show-toplevel")[0])
    # --diff-filter=d excludes deleted files; paths are relative to the repo root.
    modified = git("diff", "--name-only", "--diff-filter=d", "HEAD", "--", "*.csv", cwd=repo_root)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "*.csv", cwd=repo_root)
    return [repo_root / name for name in sorted(set(modified) | set(untracked))]


def process_csv(input_csv: Path, generator: AudioGenerator, workers: int) -> tuple[int, int, int]:
    """Generate audio for one CSV, updating it in place. Returns (generated, skipped, failed)."""
    rows = read_rows(input_csv)
    if not rows:
        print(f"{input_csv}: no data rows, skipped", file=sys.stderr)
        return 0, 0, 0

    # Only rows with at least two columns are candidates for audio generation.
    generated = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for i, row in enumerate(rows):
            if len(row) < 2:
                skipped += 1
                continue
            futures.append(pool.submit(generator.process_row, i, row[0], row[1]))

        with tqdm(total=len(rows), initial=skipped, desc=input_csv.name, unit="row") as bar:
            for future in as_completed(futures):
                index, updated_cell, error = future.result()
                english = rows[index][0]
                if error:
                    # tqdm.write prints above the bar without breaking it.
                    tqdm.write(f"row {index + 1} ({english!r}): FAILED - {error}", file=sys.stderr)
                    failed += 1
                elif updated_cell == rows[index][1]:
                    skipped += 1
                else:
                    rows[index][1] = updated_cell
                    generated += 1
                bar.update(1)

    if generated:
        write_rows(input_csv, rows)
    return generated, skipped, failed


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        sys.exit(f"error: {name} is not set (check your .env file)")
    return value


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            f"number of parallel workers (default: {DEFAULT_WORKERS}). "
            "Keep this at/below your ElevenLabs concurrent-request limit."
        ),
    )
    args = parser.parse_args()

    input_csvs = changed_csv_files()
    if not input_csvs:
        print("No changed or untracked CSV files found in git; nothing to do.")
        return 0

    media_dir = Path(env("ANKI_MEDIA_DIR"))
    if not media_dir.is_dir():
        sys.exit(f"error: Anki media folder not found: {media_dir}")

    client = ElevenLabs(api_key=env("ELEVENLABS_API_KEY"))
    generator = AudioGenerator(
        client=client,
        media_dir=media_dir,
        voice_id=env("ELEVENLABS_VOICE_ID"),
        model_id=env("ELEVENLABS_MODEL_ID"),
    )

    generated = skipped = failed = 0
    for input_csv in input_csvs:
        g, s, f = process_csv(input_csv, generator, args.workers)
        generated, skipped, failed = generated + g, skipped + s, failed + f

    print(f"\nDone. generated={generated} skipped={skipped} failed={failed}")
    print(f"Audio written to: {media_dir}")
    print("Re-import the updated CSVs into Anki to attach the audio.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
