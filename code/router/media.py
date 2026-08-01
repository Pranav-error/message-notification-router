"""Turning local media files into something a model can read.

Images become data-URI content blocks. Voice notes get transcribed once and
cached to disk, because transcription is the slowest and least interesting
part of a re-run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
AUDIO_SUFFIXES = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".aac", ".mp4"}

# Gateway image inputs are size-sensitive; anything larger gets downscaled.
MAX_IMAGE_BYTES = 4 * 1024 * 1024


MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def sniff_media_type(data: bytes) -> str | None:
    """Identify an image from its magic bytes, ignoring the file extension."""
    for signature, mime in MAGIC:
        if data.startswith(signature):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "unknown"


def image_block(path: Path) -> dict[str, Any]:
    """An image content block for a local file.

    Emitted in Anthropic's shape; the gateway backend converts it to a data
    URI on the way out (see `llm._to_openai`).
    """
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        data = _downscale(path, data)
    # Several files in this corpus are PNGs behind a .jpg extension, and the
    # API validates the declared media type against the actual bytes.
    mime = sniff_media_type(data) or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": base64.b64encode(data).decode(),
        },
    }


def _downscale(path: Path, data: bytes) -> bytes:
    try:
        import io

        from PIL import Image
    except ImportError:
        # Without Pillow we send the original and let the gateway decide.
        return data

    image = Image.open(io.BytesIO(data))
    image.thumbnail((1568, 1568))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def audio_block(path: Path) -> dict[str, Any]:
    """An audio content block for models that accept inline audio."""
    fmt = path.suffix.lower().lstrip(".")
    if fmt in {"m4a", "aac"}:
        fmt = "mp4"
    encoded = base64.b64encode(path.read_bytes()).decode()
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": fmt}}
