#!/usr/bin/env python3
"""Evaluation harness.

Three things it does:

    python code/evaluation/main.py                       # score on the 30 labelled samples
    python code/evaluation/main.py --compare a,b,c       # same harness across models
    python code/evaluation/main.py --validate output.csv # submission contract check

Scoring mirrors what the graders said they look at: action, message_type,
reason usefulness/consistency, evidence relevance, and confidence calibration.
`action` carries the most weight because it is the actual product decision, and
the mute/notify confusions are reported separately because they are the two
failures that matter asymmetrically — a missed notify is a missed emergency, a
wrong notify is the noise the whole system exists to remove.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from router import reasons, router  # noqa: E402
from router.data import Dataset  # noqa: E402
from router.llm import Client  # noqa: E402
from router.understanding import MediaIndex, Transcriber  # noqa: E402

ACTIONS = ["notify", "digest", "mute"]
RISK_TYPES = {"scam", "spam"}
OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]


def _split_evidence(value: str | None) -> set[str]:
    if not value or str(value).strip().lower() == "none":
        return set()
    return {p.strip() for p in str(value).split(";") if p.strip()}


def score(gold: list[dict], pred: dict[str, dict]) -> dict:
    """Compare predictions against labelled rows."""
    n = len(gold)
    action_hits = type_hits = both_hits = 0
    confusion: Counter = Counter()
    per_action = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    evidence_precision: list[float] = []
    evidence_recall: list[float] = []
    evidence_supplied = 0
    canonical_reasons = 0
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    brier = 0.0

    canonical_texts = {entry["text"] for entry in reasons.REASONS.values()}
    unsafe_missed: list[str] = []
    false_interrupts: list[str] = []
    missed_interrupts: list[str] = []

    for row in gold:
        mid = row["message_id"]
        p = pred.get(mid)
        if not p:
            per_action[row["action"]]["fn"] += 1
            continue

        g_action, p_action = row["action"], p["action"]
        g_type, p_type = row["message_type"], p["message_type"]
        correct = g_action == p_action

        action_hits += correct
        type_hits += g_type == p_type
        both_hits += correct and g_type == p_type
        confusion[(g_action, p_action)] += 1

        if correct:
            per_action[g_action]["tp"] += 1
        else:
            per_action[p_action]["fp"] += 1
            per_action[g_action]["fn"] += 1
            if g_action == "mute" and g_type in RISK_TYPES:
                unsafe_missed.append(mid)
            if p_action == "notify" and g_action != "notify":
                false_interrupts.append(mid)
            if g_action == "notify" and p_action != "notify":
                missed_interrupts.append(mid)

        gold_ev = _split_evidence(row.get("evidence_message_ids"))
        pred_ev = _split_evidence(p.get("evidence_message_ids"))
        if pred_ev:
            evidence_supplied += 1
        if gold_ev or pred_ev:
            overlap = len(gold_ev & pred_ev)
            evidence_precision.append(overlap / len(pred_ev) if pred_ev else 0.0)
            evidence_recall.append(overlap / len(gold_ev) if gold_ev else 0.0)

        if (p.get("reason") or "").strip() in canonical_texts:
            canonical_reasons += 1

        conf = float(p.get("confidence") or 0)
        (conf_correct if correct else conf_wrong).append(conf)
        brier += (conf - (1.0 if correct else 0.0)) ** 2

    def prf(bucket: dict) -> tuple[float, float, float]:
        tp, fp, fn = bucket["tp"], bucket["fp"], bucket["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731

    return {
        "n": n,
        "action_accuracy": action_hits / n if n else 0.0,
        "message_type_accuracy": type_hits / n if n else 0.0,
        "exact_match": both_hits / n if n else 0.0,
        "per_action": {a: dict(zip(("precision", "recall", "f1"), prf(per_action[a]))) for a in ACTIONS},
        "confusion": {f"{g}->{p}": c for (g, p), c in sorted(confusion.items())},
        "evidence_precision": mean(evidence_precision),
        "evidence_recall": mean(evidence_recall),
        "evidence_supplied_rate": evidence_supplied / n if n else 0.0,
        "canonical_reason_rate": canonical_reasons / n if n else 0.0,
        "confidence_when_correct": mean(conf_correct),
        "confidence_when_wrong": mean(conf_wrong),
        "brier_score": brier / n if n else 0.0,
        "missed_risky_messages": unsafe_missed,
        "false_interrupts": false_interrupts,
        "missed_interrupts": missed_interrupts,
    }


def report(name: str, s: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"rows scored                {s['n']}")
    print(f"action accuracy            {s['action_accuracy']:.1%}")
    print(f"message_type accuracy      {s['message_type_accuracy']:.1%}")
    print(f"both correct               {s['exact_match']:.1%}")
    print("\nper-action (precision / recall / f1)")
    for action in ACTIONS:
        m = s["per_action"][action]
        print(f"  {action:7s} {m['precision']:.2f} / {m['recall']:.2f} / {m['f1']:.2f}")
    print("\nconfusion (gold -> predicted)")
    for key, count in s["confusion"].items():
        marker = "   " if key.split("->")[0] == key.split("->")[1] else " * "
        print(f" {marker}{key:20s} {count}")
    print("\nevidence")
    print(f"  precision vs gold ids    {s['evidence_precision']:.2f}")
    print(f"  recall vs gold ids       {s['evidence_recall']:.2f}")
    print(f"  rows citing evidence     {s['evidence_supplied_rate']:.1%}")
    print("\nreason + calibration")
    print(f"  canonical reason rate    {s['canonical_reason_rate']:.1%}")
    print(f"  confidence when correct  {s['confidence_when_correct']:.3f}")
    print(f"  confidence when wrong    {s['confidence_when_wrong']:.3f}")
    print(f"  brier score (lower=good) {s['brier_score']:.3f}")
    if s["missed_risky_messages"]:
        print(f"\n  !! risky messages not muted: {', '.join(s['missed_risky_messages'])}")
    if s["false_interrupts"]:
        print(f"  !  false interrupts: {', '.join(s['false_interrupts'])}")
    if s["missed_interrupts"]:
        print(f"  !  missed interrupts: {', '.join(s['missed_interrupts'])}")


def run_split(dataset: Dataset, model: str, vision: str, asr: str, workers: int,
              use_media: bool) -> dict[str, dict]:
    cache_root = REPO_ROOT / ".cache"
    client = Client(model=model, cache_dir=cache_root / "llm")
    media_index = None
    if use_media:
        media_index = MediaIndex(
            vision=Client(model=vision, cache_dir=cache_root / "llm"),
            transcriber=Transcriber(asr),
            cache_path=cache_root / "media_index.json",
        )
    messages = dataset.messages("sample_messages")
    decisions = router.route_all(dataset, messages, client, media_index, workers=workers)
    return {d.message_id: d.as_output_row() for d in decisions}


def validate(output_path: Path, dataset: Dataset) -> int:
    """Check the submission contract before uploading."""
    problems: list[str] = []
    if not output_path.exists():
        print(f"missing {output_path}")
        return 1

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else []

    if header != OUTPUT_COLUMNS:
        problems.append(f"columns are {header}, expected {OUTPUT_COLUMNS}")

    expected = [m["message_id"] for m in dataset.messages("messages")]
    got = [r["message_id"] for r in rows]
    if len(got) != len(expected):
        problems.append(f"{len(got)} rows, expected {len(expected)}")
    if set(got) != set(expected):
        missing = set(expected) - set(got)
        extra = set(got) - set(expected)
        if missing:
            problems.append(f"missing ids: {sorted(missing)[:5]}")
        if extra:
            problems.append(f"unexpected ids: {sorted(extra)[:5]}")
    if len(set(got)) != len(got):
        problems.append("duplicate message_id rows")

    valid_types = {
        "personal", "urgent", "event", "payment", "business_update", "promotion",
        "greeting", "forward", "spam", "scam", "unknown",
    }
    for r in rows:
        if r["action"] not in ACTIONS:
            problems.append(f"{r['message_id']}: bad action {r['action']!r}")
        if r["message_type"] not in valid_types:
            problems.append(f"{r['message_id']}: bad message_type {r['message_type']!r}")
        try:
            c = float(r["confidence"])
            if not 0.0 <= c <= 1.0:
                problems.append(f"{r['message_id']}: confidence {c} out of range")
        except ValueError:
            problems.append(f"{r['message_id']}: non-numeric confidence {r['confidence']!r}")
        if not (r["reason"] or "").strip():
            problems.append(f"{r['message_id']}: empty reason")
        if not (r["evidence_message_ids"] or "").strip():
            problems.append(f"{r['message_id']}: empty evidence (use 'none')")

    if problems:
        print(f"FAILED ({len(problems)} problems)")
        for p in problems[:25]:
            print(f"  - {p}")
        return 1

    dist = Counter(r["action"] for r in rows)
    print(f"OK  {len(rows)} rows, columns valid, ids match dataset/messages.csv")
    print(f"    action distribution: {dict(dist)}")
    print(f"    message_type spread: {dict(Counter(r['message_type'] for r in rows))}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate the notification router.")
    p.add_argument("--dataset", default=str(REPO_ROOT / "dataset"))
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--vision-model", default="claude-opus-5")
    p.add_argument("--asr-model", default="small")
    p.add_argument("--compare", default="", help="comma-separated models to score side by side")
    p.add_argument("--predictions", default="", help="score an existing CSV instead of routing")
    p.add_argument("--validate", default="", help="validate a submission CSV and exit")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-media", action="store_true")
    p.add_argument("--json-out", default="")
    args = p.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "code"))
    from main import load_env  # noqa: E402

    load_env(REPO_ROOT / ".env")
    dataset = Dataset.load(args.dataset)

    if args.validate:
        return validate(Path(args.validate), dataset)

    gold = dataset.messages("sample_messages")
    results: dict[str, dict] = {}

    if args.predictions:
        with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
            pred = {r["message_id"]: r for r in csv.DictReader(handle)}
        results[args.predictions] = score(gold, pred)
    else:
        models = [m.strip() for m in args.compare.split(",") if m.strip()] or [args.model]
        for model in models:
            pred = run_split(
                dataset, model, args.vision_model, args.asr_model,
                args.workers, not args.no_media,
            )
            results[model] = score(gold, pred)

    for name, s in results.items():
        report(name, s)

    if len(results) > 1:
        print("\n=== comparison ===")
        print(f"{'model':38s} {'action':>8s} {'type':>8s} {'exact':>8s} {'brier':>8s}")
        for name, s in sorted(results.items(), key=lambda kv: -kv[1]["action_accuracy"]):
            print(
                f"{name:38s} {s['action_accuracy']:>7.1%} {s['message_type_accuracy']:>7.1%} "
                f"{s['exact_match']:>7.1%} {s['brier_score']:>8.3f}"
            )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
