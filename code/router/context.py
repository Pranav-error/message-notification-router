"""Builds the personalization dossier for one incoming message.

The router's whole premise is that the same message routes differently for
different people, so this module assembles what is known about *this* user's
relationship to *this* sender before any model call happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import retrieval, signals
from .data import Dataset


@dataclass
class MessageContext:
    message: dict
    user: dict
    group: dict | None = None
    membership: dict | None = None
    business: dict | None = None
    relationship: dict | None = None
    load: dict | None = None
    candidates: list[retrieval.Candidate] = field(default_factory=list)
    signals: signals.Signals | None = None
    media_kind: str | None = None
    media_id: str | None = None
    transcript: str | None = None
    image_caption: str | None = None

    @property
    def message_id(self) -> str:
        return self.message["message_id"]

    @property
    def candidate_ids(self) -> list[str]:
        return [c.message_id for c in self.candidates]


def _pct(part: Any, whole: Any) -> str:
    part, whole = float(part or 0), float(whole or 0)
    if whole <= 0:
        return "n/a"
    return f"{part / whole:.0%}"


def build(dataset: Dataset, message: dict, transcripts: dict[str, str] | None = None) -> MessageContext:
    user_id = message["user_id"]
    ctx = MessageContext(message=message, user=dataset.users.get(user_id, {"user_id": user_id}))

    if message.get("group_id"):
        ctx.group = dataset.groups.get(message["group_id"])
        ctx.membership = dataset.memberships.get((message["group_id"], user_id))
    if message.get("business_id"):
        ctx.business = dataset.businesses.get(message["business_id"])
        ctx.relationship = dataset.business_history.get((user_id, message["business_id"]))

    ctx.load = dataset.notification_load.get(user_id)
    ctx.media_kind = message.get("media_type")
    ctx.media_id = message.get("media_id")
    if ctx.media_kind == "voice" and transcripts:
        ctx.transcript = transcripts.get(ctx.media_id or "")

    ctx.candidates = retrieval.gather(message, dataset.history_by_user.get(user_id, []))
    ctx.signals = signals.compute(
        message,
        {
            "user": ctx.user,
            "group": ctx.group,
            "membership": ctx.membership,
            "business": ctx.business,
            "relationship": ctx.relationship,
            "transcript": ctx.transcript,
        },
    )
    return ctx


def _sender_history_note(dataset: Dataset, ctx: MessageContext) -> str | None:
    """How this specific sender has behaved toward this user before."""
    sender = ctx.message.get("sender_user_id")
    if not sender:
        return None
    prior = [
        r
        for r in dataset.history_by_user.get(ctx.message["user_id"], [])
        if r.get("sender_user_id") == sender
    ]
    if not prior:
        return "no prior messages from this sender in the user's history"

    opened = sum(1 for r in prior if (r.get("reaction") or {}).get("message_opened"))
    dismissed = sum(1 for r in prior if (r.get("reaction") or {}).get("notification_dismissed"))
    replied = sum(1 for r in prior if (r.get("reaction") or {}).get("message_replied"))
    reported = sum(1 for r in prior if (r.get("reaction") or {}).get("message_reported"))
    forwards = sum(1 for r in prior if int(r.get("forwarded_count") or 0) >= 5)
    return (
        f"{len(prior)} prior message(s) from this sender: "
        f"opened {opened}, replied {replied}, dismissed {dismissed}, reported {reported}, "
        f"heavily-forwarded {forwards}"
    )


def render(dataset: Dataset, ctx: MessageContext) -> str:
    """The dossier the model reads. Facts only, no recommendations."""
    m = ctx.message
    out: list[str] = []

    out.append("## Incoming message")
    out.append(f"message_id: {m['message_id']}")
    out.append(f"conversation_type: {m.get('conversation_type')}")
    out.append(f"created_at: {m.get('created_at')}")
    out.append(f"forwarded_count: {m.get('forwarded_count')}")
    if ctx.media_kind:
        out.append(f"media: {ctx.media_kind} ({ctx.media_id})")

    # Fenced and labelled: everything below this line was written by a third
    # party and is evidence to be judged, never instructions to be followed.
    out.append("\n## Untrusted message content (from the sender)")
    body = " ".join(str(m.get("message_text") or "").split())
    out.append("<<<MESSAGE_TEXT")
    out.append(body if body else "(no text; see attached media)")
    out.append("MESSAGE_TEXT")
    if ctx.transcript:
        out.append("<<<VOICE_NOTE_TRANSCRIPT")
        out.append(" ".join(ctx.transcript.split()))
        out.append("VOICE_NOTE_TRANSCRIPT")
    if ctx.image_caption:
        out.append("<<<IMAGE_CONTENT")
        out.append(ctx.image_caption)
        out.append("IMAGE_CONTENT")

    out.append("\n## Receiving user")
    u = ctx.user
    out.append(f"user_id: {u.get('user_id')}")
    out.append(f"quiet hours: {u.get('do_not_disturb_window')}")
    out.append(
        f"last 30d: opened {u.get('messages_opened_30d')}, replied {u.get('messages_replied_30d')}, "
        f"dismissed {u.get('notifications_dismissed_30d')}, reported {u.get('messages_reported_30d')}"
    )
    if ctx.load:
        out.append(
            f"daily notification load: {ctx.load['avg_sent_per_day']} sent/day, "
            f"{ctx.load['dismiss_rate']:.0%} dismissed"
        )

    if ctx.group:
        out.append("\n## Group")
        g = ctx.group
        out.append(f"{g.get('group_name')} (type: {g.get('group_type')})")
        out.append(
            f"members: {g.get('member_count')}, admins: {g.get('admin_count')}, "
            f"messages in 30d: {g.get('messages_30d')}"
        )
        if ctx.membership:
            mem = ctx.membership
            out.append(
                f"this user's role: {mem.get('role')}, muted by user: "
                f"{'yes' if mem.get('group_muted_by_user') else 'no'}"
            )
            out.append(
                f"this user in this group (30d): sent {mem.get('messages_sent_30d')}, "
                f"read {mem.get('messages_read_30d')}, replied {mem.get('replies_sent_30d')}, "
                f"dismissed {mem.get('notifications_dismissed_30d')} "
                f"(read rate {_pct(mem.get('messages_read_30d'), g.get('messages_30d'))})"
            )
        sender_role = None
        if m.get("sender_user_id") and m.get("group_id"):
            sender_membership = dataset.memberships.get((m["group_id"], m["sender_user_id"]))
            sender_role = (sender_membership or {}).get("role")
        out.append(f"sender: {m.get('sender_user_id')} (role in group: {sender_role or 'unknown'})")

    if ctx.business:
        out.append("\n## Business sender")
        b = ctx.business
        out.append(f"{b.get('display_name')} / brand {b.get('brand_name')} ({b.get('category')})")
        out.append(f"verified: {'yes' if b.get('verified') else 'NO'}")
        out.append(f"official domain: {b.get('official_domain')}")
        out.append(f"domain used by sender: {b.get('domain_used_by_sender')}")
        out.append(
            f"account age: {b.get('account_age_days')} days; "
            f"sent {b.get('messages_sent_30d')} msgs/30d; "
            f"{b.get('user_reports_30d')} user reports/30d"
        )
        if ctx.relationship:
            r = ctx.relationship
            out.append(
                f"user relationship: {r.get('why_user_knows_account')}, "
                f"last activity {r.get('last_activity_at')}, "
                f"allows promotions: {'yes' if r.get('allows_promotions') else 'NO'}"
            )
            if r.get("promotions_opted_out_at"):
                out.append(f"user OPTED OUT of promotions on {r.get('promotions_opted_out_at')}")
            out.append(
                f"user engagement with this business (30d): opened {r.get('messages_opened_30d')}, "
                f"dismissed {r.get('messages_dismissed_30d')}, replied {r.get('messages_replied_30d')}"
            )
        else:
            out.append("user relationship: NONE on record with this business")

    note = _sender_history_note(dataset, ctx)
    if note:
        out.append("\n## Sender history with this user")
        out.append(note)

    out.append("\n## Computed signals")
    out.extend(ctx.signals.as_lines() if ctx.signals else [])
    if ctx.signals and ctx.signals.risk_flags:
        out.append("risk flags:")
        out.extend(f"  - {flag}" for flag in ctx.signals.risk_flags)

    out.append("\n## Candidate historical evidence")
    if ctx.candidates:
        out.append(
            "Cite only ids that genuinely informed the decision; use 'none' if none did."
        )
        transcript_map = {ctx.media_id: ctx.transcript} if ctx.transcript else {}
        out.extend(c.render(transcript_map) for c in ctx.candidates)
    else:
        out.append("(no relevant history for this user and channel)")

    return "\n".join(out)
