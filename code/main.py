#!/usr/bin/env python3
"""Message Notification Router — prediction entry point.

    python code/main.py                      # route dataset/messages.csv -> output.csv
    python code/main.py --split sample_messages --out predictions/sample.csv
    python code/main.py --model openai/gpt-5.4 --limit 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from router import cascade, router  # noqa: E402
from router.data import Dataset  # noqa: E402
from router.llm import Client  # noqa: E402
from router.understanding import MediaIndex, Transcriber  # noqa: E402

OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_VISION_MODEL = "claude-opus-5"
DEFAULT_ASR_MODEL = "small"


def load_env(path: Path) -> None:
    """Minimal .env reader so secrets stay out of the repo and the shell history."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route WhatsApp messages to notify/digest/mute.")
    p.add_argument("--dataset", default=str(REPO_ROOT / "dataset"))
    p.add_argument("--split", default="messages", choices=["messages", "sample_messages"])
    p.add_argument("--out", default=str(REPO_ROOT / "output.csv"))
    p.add_argument("--debug-out", default=str(REPO_ROOT / "predictions" / "debug.jsonl"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL, help="local Whisper size")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="route only the first N messages")
    p.add_argument("--no-media", action="store_true", help="skip image/voice understanding")
    p.add_argument("--no-cascade", action="store_true",
                   help="disable tier-1 rules; send every message to the model")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    load_env(REPO_ROOT / ".env")

    dataset = Dataset.load(args.dataset)
    messages = dataset.messages(args.split)
    if args.limit:
        messages = messages[: args.limit]

    cache_root = REPO_ROOT / ".cache"
    client = Client(model=args.model, cache_dir=cache_root / "llm")
    media_index = None
    if not args.no_media:
        media_index = MediaIndex(
            vision=Client(model=args.vision_model, cache_dir=cache_root / "llm"),
            transcriber=Transcriber(args.asr_model),
            cache_path=cache_root / "media_index.json",
        )

    print(f"routing {len(messages)} messages with {args.model}", file=sys.stderr)
    counter = {"n": 0}
    lock = threading.Lock()

    def progress(_decision) -> None:
        with lock:
            counter["n"] += 1
            done = counter["n"]
        if done % 10 == 0 or done == len(messages):
            print(f"  {done}/{len(messages)}", file=sys.stderr)

    stats = cascade.CascadeStats()
    decisions = router.route_all(
        dataset, messages, client, media_index, workers=args.workers, on_done=progress,
        stats=stats, use_cascade=not args.no_cascade,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(router.to_records(decisions))

    debug_path = Path(args.debug_out)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8") as handle:
        for record in router.debug_records(decisions):
            handle.write(json.dumps(record) + "\n")

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1
    overridden = sum(1 for d in decisions if d.overrides)

    print(f"\nwrote {out_path} ({len(decisions)} rows)", file=sys.stderr)
    print(f"action distribution: {counts}", file=sys.stderr)
    print(f"safety-floor overrides: {overridden}", file=sys.stderr)
    print(f"model usage: {client.usage.summary()}", file=sys.stderr)
    if not args.no_cascade:
        print("\n" + stats.report(), file=sys.stderr)
        stats_path = debug_path.parent / "cascade_stats.json"
        stats_path.write_text(json.dumps(stats.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
