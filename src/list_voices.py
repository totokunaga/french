"""Print the French voices available in your ElevenLabs account.

Use this to pick a `voice_id` for ELEVENLABS_VOICE_ID in .env.
"""

import os

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


def main() -> None:
    load_dotenv()
    client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
    for voice in client.voices.get_all().voices:
        labels = voice.labels or {}
        if labels.get("language") != "fr":
            continue
        accent = labels.get("accent", "?")
        gender = labels.get("gender", "?")
        print(f"{voice.voice_id}  {voice.name}  ({accent}, {gender})")


if __name__ == "__main__":
    main()
