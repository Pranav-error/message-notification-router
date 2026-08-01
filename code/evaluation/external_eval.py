#!/usr/bin/env python3
"""External evaluation on a public, human-labelled corpus.

The generalization suite tests messages I wrote. This tests messages nobody
involved in this project wrote: the UCI SMS Spam Collection (Almeida, Gomez
Hidalgo & Yamakami, 2011) — 5,574 real SMS messages labelled `spam` or `ham` by
human annotators, downloaded at runtime.

    python code/evaluation/external_eval.py --limit 60

WHAT THIS DOES AND DOES NOT TEST
--------------------------------
It tests the **content-safety path only**, and the framing has to be honest
about that:

- SMS carries no personalization context. There is no sender relationship, no
  engagement history, no quiet hours, no opt-out state. The dossier is nearly
  empty, so the router is running without the signal it is actually built
  around. This is the router with one hand tied behind its back.
- The label space differs. `spam` maps to `mute`; `ham` maps to "not mute"
  (either notify or digest). Only the mute/not-mute axis is scorable, so a ham
  message routed to digest and one routed to notify both count as correct.
- The domain differs. This corpus is UK SMS from around 2011 — premium-rate
  short codes, "claim your prize", ringtone subscriptions. The challenge corpus
  is 2026 Indian WhatsApp — UPI, KYC, OTP, society maintenance. Scam *idiom*
  therefore differs sharply, which makes this genuinely out-of-distribution
  rather than a restatement of the same test.

Each message is routed as if it arrived from an unknown number, which is the
honest analogue of an SMS: `conversation_type=personal`, a sender with no
history. Nothing in the pipeline was tuned against these results.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from main import load_env  # noqa: E402
from router import cascade, router  # noqa: E402
from router.data import Dataset  # noqa: E402
from router.llm import Client  # noqa: E402
from router.understanding import MediaIndex, Transcriber  # noqa: E402

SOURCE = (
    "https://raw.githubusercontent.com/mohitgupta-omg/"
    "Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
)


def load_corpus(cache: Path) -> list[tuple[str, str]]:
    """Download (once) and parse the SMS corpus into (label, text) pairs."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {SOURCE}", file=sys.stderr)
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            cache.write_bytes(response.read())

    raw = cache.read_bytes().decode("latin-1")
    rows = []
    for row in csv.reader(io.StringIO(raw)):
        if len(row) < 2 or row[0] not in {"ham", "spam"}:
            continue
        text = row[1].strip()
        if len(text) > 20:  # drop fragments too short to judge
            rows.append((row[0], text))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Score the router on a public SMS corpus.")
    p.add_argument("--dataset", default=str(REPO_ROOT / "dataset"))
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--limit", type=int, default=60, help="total messages (balanced)")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--user", default="u_003", help="receiving user for the dossier")
    args = p.parse_args()

    load_env(REPO_ROOT / ".env")
    corpus = load_corpus(REPO_ROOT / ".cache" / "sms_spam.csv")
    counts = Counter(label for label, _ in corpus)
    print(f"corpus: {len(corpus)} messages  {dict(counts)}")

    random.seed(args.seed)
    per_class = args.limit // 2
    sample: list[tuple[str, str]] = []
    for label in ("spam", "ham"):
        members = [x for x in corpus if x[0] == label]
        random.shuffle(members)
        sample.extend(members[:per_class])
    random.shuffle(sample)
    print(f"evaluating {len(sample)} messages ({per_class} spam / {per_class} ham)\n")

    dataset = Dataset.load(args.dataset)
    cache_root = REPO_ROOT / ".cache"
    client = Client(model=args.model, cache_dir=cache_root / "llm")
    media_index = MediaIndex(
        vision=Client(model=args.model, cache_dir=cache_root / "llm"),
        transcriber=Transcriber("small"),
        cache_path=cache_root / "media_index.json",
    )
    stats = cascade.CascadeStats()

    confusion: Counter = Counter()
    missed_spam, muted_ham = [], []
    for index, (label, text) in enumerate(sample):
        message = {
            "message_id": f"sms_{index:03d}",
            "user_id": args.user,
            "conversation_type": "personal",
            "group_id": None,
            "business_id": None,
            # An unknown number: no relationship, no history, which is exactly
            # what an unsolicited SMS is.
            "sender_user_id": f"sms_sender_{index:03d}",
            "created_at": "2026-08-02 14:30",
            "message_text": text,
            "media_type": None,
            "media_id": None,
            "forwarded_count": 0,
        }
        decision = router.route_one(dataset, message, client, media_index, stats=stats)
        muted = decision.action == "mute"
        should_mute = label == "spam"
        confusion[(label, "mute" if muted else "not-mute")] += 1
        if should_mute and not muted:
            missed_spam.append((text, decision))
        if not should_mute and muted:
            muted_ham.append((text, decision))

    n = len(sample)
    correct = confusion[("spam", "mute")] + confusion[("ham", "not-mute")]
    spam_n = confusion[("spam", "mute")] + confusion[("spam", "not-mute")]
    ham_n = confusion[("ham", "mute")] + confusion[("ham", "not-mute")]

    print(f"overall agreement (mute vs not-mute)  {correct}/{n} = {correct/n:.1%}")
    print(f"  spam correctly muted (recall)       {confusion[('spam','mute')]}/{spam_n} "
          f"= {confusion[('spam','mute')]/max(spam_n,1):.1%}")
    print(f"  ham correctly not muted (specificity) {confusion[('ham','not-mute')]}/{ham_n} "
          f"= {confusion[('ham','not-mute')]/max(ham_n,1):.1%}")
    print(f"\ncascade escalation: {stats.summary()['escalation_rate']:.1%}")
    print(f"model usage: {client.usage.summary()}")

    if missed_spam:
        print(f"\n--- spam the router did NOT mute ({len(missed_spam)}) ---")
        for text, d in missed_spam[:10]:
            print(f"  [{d.action}/{d.message_type}] {' '.join(text.split())[:100]}")
    if muted_ham:
        print(f"\n--- legitimate messages the router muted ({len(muted_ham)}) ---")
        for text, d in muted_ham[:10]:
            print(f"  [{d.message_type}] {' '.join(text.split())[:100]}")

    print("\nNote: no personalization context exists for SMS, so this exercises the "
          "content-safety\npath only — the router without the history it is built around.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
