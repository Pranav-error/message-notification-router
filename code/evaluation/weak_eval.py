#!/usr/bin/env python3
"""Behavioural evaluation on weak labels derived from real user reactions.

Only 30 messages carry gold labels, which is thin evidence for a 110-row
submission. But `message_events.csv` records what 412 users *actually did* with
412 historical messages — opened, replied, dismissed, muted the chat, reported
it. Those reactions are a noisy proxy for what the router should have done:

    reported / muted the chat afterwards      -> should have been muted
    dismissed and never opened                -> should have been muted
    opened and replied quickly                -> should have interrupted (notify)
    opened, no reply, not dismissed           -> could have waited (digest)

This is a *proxy*, not ground truth — a user can dismiss something that mattered,
and reaction time depends on when the notification actually surfaced. So the
headline number here is **agreement**, not accuracy, and the mute/not-mute split
is reported separately because it is the axis the reactions describe most
reliably.

    python code/evaluation/weak_eval.py --limit 80

LEAKAGE: each message under test lives in the very history the router reads as
evidence. Left alone, the router could see the target's own recorded reaction
and trivially "predict" it. Every target is therefore removed from its own
history before routing — see `_without`.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from main import load_env  # noqa: E402
from router import cascade, router  # noqa: E402
from router.data import Dataset  # noqa: E402
from router.llm import Client  # noqa: E402
from router.understanding import MediaIndex, Transcriber  # noqa: E402

FAST_REPLY_MINUTES = 15


def weak_label(reaction: dict | None) -> str | None:
    """Derive the action the user's own behaviour implies, or None if unclear."""
    if not reaction:
        return None

    if reaction.get("message_reported") or reaction.get("muted_after_message"):
        return "mute"

    opened = bool(reaction.get("message_opened"))
    replied = bool(reaction.get("message_replied"))
    dismissed = bool(reaction.get("notification_dismissed"))
    minutes = reaction.get("reaction_time_minutes")

    if dismissed and not opened:
        return "mute"
    if opened and replied and minutes is not None and float(minutes) <= FAST_REPLY_MINUTES:
        return "notify"
    if opened and not replied and not dismissed:
        return "digest"
    return None


def _without(dataset: Dataset, message_id: str) -> Dataset:
    """A view of the dataset with one message erased from every index.

    Without this the router reads the target's own reaction as evidence about
    itself, and the evaluation measures nothing.
    """
    by_user = {
        user: [r for r in rows if r["message_id"] != message_id]
        for user, rows in dataset.history_by_user.items()
    }
    by_id = {k: v for k, v in dataset.history_by_id.items() if k != message_id}
    return replace(dataset, history_by_user=by_user, history_by_id=by_id)


def main() -> int:
    p = argparse.ArgumentParser(description="Score the router against observed user behaviour.")
    p.add_argument("--dataset", default=str(REPO_ROOT / "dataset"))
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    load_env(REPO_ROOT / ".env")
    dataset = Dataset.load(args.dataset)

    # Build the labelled pool from history messages whose reaction is decisive.
    pool = []
    for record in dataset.history_by_id.values():
        label = weak_label(record.get("reaction"))
        if label:
            pool.append((record, label))

    counts = Counter(label for _, label in pool)
    print(f"weak-labelled pool: {len(pool)} of {len(dataset.history_by_id)} history messages")
    print(f"  distribution: {dict(counts)}")

    # Stratify so a majority class cannot carry the score.
    random.seed(args.seed)
    per_class = max(1, args.limit // max(len(counts), 1))
    sample: list[tuple[dict, str]] = []
    for label in counts:
        members = [x for x in pool if x[1] == label]
        random.shuffle(members)
        sample.extend(members[:per_class])
    random.shuffle(sample)
    print(f"  evaluating a stratified sample of {len(sample)}")

    cache_root = REPO_ROOT / ".cache"
    client = Client(model=args.model, cache_dir=cache_root / "llm")
    media_index = MediaIndex(
        vision=Client(model=args.model, cache_dir=cache_root / "llm"),
        transcriber=Transcriber("small"),
        cache_path=cache_root / "media_index.json",
    )
    stats = cascade.CascadeStats()

    agree = mute_agree = 0
    confusion: Counter = Counter()
    for record, label in sample:
        held_out = _without(dataset, record["message_id"])
        decision = router.route_one(held_out, record, client, media_index, stats=stats)
        confusion[(label, decision.action)] += 1
        agree += decision.action == label
        mute_agree += (decision.action == "mute") == (label == "mute")

    n = len(sample)
    print(f"\n3-way agreement with observed behaviour   {agree / n:.1%}  ({agree}/{n})")
    print(f"mute vs not-mute agreement                {mute_agree / n:.1%}  ({mute_agree}/{n})")
    print("\nconfusion (user behaviour -> router)")
    for (behaviour, action), count in sorted(confusion.items()):
        marker = "   " if behaviour == action else " * "
        print(f" {marker}{behaviour:7s} -> {action:7s}  {count}")
    print(f"\ncascade: {stats.summary()['escalation_rate']:.1%} escalated")
    print(f"model usage: {client.usage.summary()}")
    print(
        "\nThese are proxy labels: a user dismissing a message does not prove it "
        "should have been muted.\nRead the mute/not-mute row as the reliable one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
