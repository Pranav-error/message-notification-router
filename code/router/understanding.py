"""Media understanding: posters and screenshots in, voice notes in.

Both are resolved to text *once* per media file and cached to disk, then folded
into the dossier. Doing it as a separate pass rather than attaching raw media to
the routing call means the same poster costs one vision call no matter how many
users received it, and the routing model sees image text and speech in exactly
the same form as ordinary message text.

Images go to a vision model. Voice notes are transcribed **locally** with
Whisper: the Anthropic API has no audio input, and running ASR on-device keeps
the pipeline working on a text-and-vision-only provider, costs nothing per call,
and keeps voice content off a third-party service.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import media
from .llm import Client
from .prompts import IMAGE_PROMPT, MEDIA_SCHEMA

DEFAULT_ASR_MODEL = "small"


class Transcriber:
    """Local Whisper ASR, loaded lazily so a text-only run never pays for it."""

    def __init__(self, model_size: str = DEFAULT_ASR_MODEL) -> None:
        self.model_size = model_size
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(
                    self.model_size, device="cpu", compute_type="int8"
                )
        return self._model

    def transcribe(self, path: Path) -> dict:
        model = self._ensure_model()
        segments, info = model.transcribe(str(path), beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return {
            "transcript": text,
            "language": info.language,
            "duration_seconds": round(info.duration, 1),
        }


class MediaIndex:
    """Resolves media ids to text, caching results on disk between runs."""

    def __init__(
        self,
        vision: Client,
        transcriber: Transcriber | None = None,
        cache_path: str | Path = ".cache/media_index.json",
    ) -> None:
        self.vision = vision
        self.transcriber = transcriber or Transcriber()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _save(self) -> None:
        self.cache_path.write_text(json.dumps(self._cache, indent=2, sort_keys=True))

    def _remember(self, key: str, value: dict) -> dict:
        with self._lock:
            self._cache[key] = value
            self._save()
        return value

    def describe_image(self, media_id: str, path: Path) -> dict:
        key = f"image:{media_id}"
        if key in self._cache:
            return self._cache[key]
        result = self.vision.json(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_PROMPT},
                        media.image_block(path),
                    ],
                }
            ],
            schema=MEDIA_SCHEMA,
            schema_name="image_reading",
        )
        return self._remember(key, result)

    def transcribe(self, media_id: str, path: Path) -> dict:
        key = f"audio:{media_id}"
        if key in self._cache:
            return self._cache[key]
        return self._remember(key, self.transcriber.transcribe(path))

    def resolve(self, media_id: str | None, media_kind: str | None, path: Path | None) -> dict:
        """Returns {'caption': str|None, 'transcript': str|None}."""
        if not media_id or not path or not path.exists():
            return {"caption": None, "transcript": None}

        if media_kind == "image":
            r = self.describe_image(media_id, path)
            parts = [f"type: {r.get('category')}", f"description: {r.get('description')}"]
            if (r.get("visible_text") or "").strip():
                parts.append(f"text in image: {r['visible_text'].strip()}")
            if (r.get("solicits_action") or "none").strip().lower() not in {"none", ""}:
                parts.append(f"image asks viewer to: {r['solicits_action'].strip()}")
            return {"caption": "\n".join(parts), "transcript": None}

        if media_kind == "voice":
            r = self.transcribe(media_id, path)
            text = (r.get("transcript") or "").strip()
            if not text:
                return {"caption": None, "transcript": "(voice note had no audible speech)"}
            duration = r.get("duration_seconds")
            suffix = f" ({duration}s voice note)" if duration else ""
            return {"caption": None, "transcript": f"{text}{suffix}"}

        return {"caption": None, "transcript": None}
