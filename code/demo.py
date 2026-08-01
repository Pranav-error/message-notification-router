#!/usr/bin/env python3
"""Interactive live router — type a message, watch it get routed.

    python code/demo.py --user u_001 --group group_001 --sender u_051
    python code/demo.py --user u_008 --business business_036
    python code/demo.py --user u_005 --group group_005 --image dataset/media/images/img_002.jpg

Each line you type is routed as if it had just arrived for that user, against
their real history, quiet hours and engagement record. The output shows which
tier decided, the fused posterior when tier 1 claimed it, and the evidence.

This exists because the interesting property of this router is that the same
text routes differently for different people — which is far easier to show than
to describe. Run it twice with two `--user` values and send the same message.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "code"))

from main import load_env  # noqa: E402
from router import belief, cascade, context, router  # noqa: E402
from router.data import Dataset  # noqa: E402
from router.llm import Client  # noqa: E402
from router.understanding import MediaIndex, Transcriber  # noqa: E402

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
COLOUR = {"notify": "\033[91m", "digest": "\033[93m", "mute": "\033[90m"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route messages interactively.")
    p.add_argument("--user", required=True, help="receiving user id, e.g. u_001")
    p.add_argument("--group", default=None, help="group id, if this is a group message")
    p.add_argument("--business", default=None, help="business id, if from a business")
    p.add_argument("--sender", default=None, help="sender user id")
    p.add_argument("--forwarded", type=int, default=0, help="forwarded_count")
    p.add_argument("--at", default=None, help="timestamp 'YYYY-MM-DD HH:MM' (default: now)")
    p.add_argument("--image", default=None, help="attach an image file")
    p.add_argument("--voice", default=None, help="attach a voice note")
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--dataset", default=str(REPO_ROOT / "dataset"))
    return p.parse_args()


def describe_user(dataset: Dataset, user_id: str, args) -> None:
    user = dataset.users.get(user_id)
    if not user:
        print(f"unknown user {user_id}; known ids look like u_001", file=sys.stderr)
        raise SystemExit(2)
    print(f"\n{BOLD}Routing for {user_id}{RESET}")
    print(f"  quiet hours     {user.get('do_not_disturb_window')}")
    print(
        f"  last 30d        opened {user.get('messages_opened_30d')}, "
        f"replied {user.get('messages_replied_30d')}, "
        f"dismissed {user.get('notifications_dismissed_30d')}, "
        f"reported {user.get('messages_reported_30d')}"
    )
    if args.group:
        group = dataset.groups.get(args.group) or {}
        member = dataset.memberships.get((args.group, user_id)) or {}
        print(
            f"  group           {group.get('group_name')} ({group.get('group_type')}), "
            f"muted by user: {'yes' if member.get('group_muted_by_user') else 'no'}"
        )
    if args.business:
        biz = dataset.businesses.get(args.business) or {}
        rel = dataset.business_history.get((user_id, args.business))
        print(
            f"  business        {biz.get('brand_name')} "
            f"(verified: {'yes' if biz.get('verified') else 'NO'}, "
            f"{biz.get('account_age_days')}d old, {biz.get('user_reports_30d')} reports)"
        )
        if rel:
            print(
                f"  relationship    {rel.get('why_user_knows_account')}, "
                f"promotions allowed: {'yes' if rel.get('allows_promotions') else 'NO'}"
            )
        else:
            print("  relationship    none on record")


def render(decision, ctx, elapsed: float) -> None:
    colour = COLOUR.get(decision.action, "")
    print(f"\n  {colour}{BOLD}{decision.action.upper():>7}{RESET}  {decision.message_type}")
    print(f"  {DIM}reason{RESET}      {decision.reason}")
    print(f"  {DIM}confidence{RESET}  {decision.confidence}")
    print(f"  {DIM}evidence{RESET}    {decision.evidence_message_ids}")

    tier = decision.raw.get("tier")
    if tier == 1:
        print(f"  {DIM}decided by{RESET}  tier 1 — no model call ({decision.risk_note})")
    else:
        print(f"  {DIM}decided by{RESET}  tier 2 — model")
        if decision.risk_note:
            print(f"  {DIM}note{RESET}        {decision.risk_note}")

    sig = ctx.signals
    if sig and sig.risk_flags:
        print(f"  {DIM}risk flags{RESET}  " + "; ".join(sig.risk_flags))
    print(f"  {DIM}latency{RESET}     {elapsed:.2f}s")


def main() -> int:
    args = parse_args()
    load_env(REPO_ROOT / ".env")
    dataset = Dataset.load(args.dataset)
    describe_user(dataset, args.user, args)

    cache_root = REPO_ROOT / ".cache"
    client = Client(model=args.model, cache_dir=cache_root / "llm")
    media_index = MediaIndex(
        vision=Client(model=args.model, cache_dir=cache_root / "llm"),
        transcriber=Transcriber("small"),
        cache_path=cache_root / "media_index.json",
    )

    conversation = "group" if args.group else "business" if args.business else "personal"
    media_type = "image" if args.image else "voice" if args.voice else None
    media_id = None
    if media_type:
        # Register the ad-hoc file under a synthetic id so the media index and
        # the dossier can find it exactly as they would for dataset media.
        media_id = f"live_{Path(args.image or args.voice).stem}"
        dataset.media_paths[media_id] = Path(args.image or args.voice)
        print(f"  attached        {media_type}: {args.image or args.voice}")

    print(f"\n{DIM}Type a message and press enter. Ctrl-D to quit.{RESET}")
    counter = 0
    while True:
        try:
            text = input(f"\n{BOLD}message>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text and not media_type:
            continue

        counter += 1
        message = {
            "message_id": f"live_{counter:03d}",
            "user_id": args.user,
            "conversation_type": conversation,
            "group_id": args.group,
            "business_id": args.business,
            "sender_user_id": args.sender,
            "created_at": args.at or datetime.now().strftime("%Y-%m-%d %H:%M"),
            "message_text": text or None,
            "media_type": media_type,
            "media_id": media_id,
            "forwarded_count": args.forwarded,
        }

        started = datetime.now()
        stats = cascade.CascadeStats()
        decision = router.route_one(dataset, message, client, media_index, stats=stats)
        elapsed = (datetime.now() - started).total_seconds()

        ctx = context.build(dataset, message)
        render(decision, ctx, elapsed)

        # Show the fused belief even when it did not clear the settle threshold,
        # so the escalation decision is visible rather than implicit.
        signals = belief.gather_signals(dataset, ctx)
        if signals:
            fused = belief.fuse(signals)
            # Report what actually happened, not just whether the posterior
            # cleared the bar: a message can clear it and still escalate,
            # because settling also requires the *type* to be unambiguous.
            if decision.raw.get("tier") == 1:
                marker = "settled by tier 1"
            elif fused.posterior >= belief.SETTLE_HIGH:
                marker = "over threshold, but type was arguable -> escalated"
            else:
                marker = "below threshold -> escalated"
            print(f"  {DIM}belief{RESET}      {fused.explain()}  [{marker}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
