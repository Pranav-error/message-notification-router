"""Evidence retrieval over the user's own message history.

`evidence_message_ids` is scored on whether the cited history is actually
relevant, so retrieval has to surface the *right* few candidates rather than a
generic recent-messages dump. Candidates are ranked on three things:

- channel match (same group / business / sender as the incoming message)
- lexical similarity to the incoming message
- how the user reacted, since an ignored-and-dismissed near-duplicate is the
  single most useful piece of evidence for a `mute`

Ranking is deterministic and dependency-free; the model picks which of the
shortlisted candidates it actually relied on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those", "it",
    "you", "your", "we", "our", "i", "me", "my", "they", "them", "with", "from", "by",
    "as", "so", "not", "no", "can", "will", "just", "have", "has", "had", "do", "does",
    "please", "pls", "hi", "hello", "dear", "customer", "now", "today", "get", "up",
}

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in TOKEN.findall(str(text).lower()) if t not in STOPWORDS and len(t) > 2}


def similarity(a: str | None, b: str | None) -> float:
    """Jaccard overlap on content words."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse(stamp: Any) -> datetime | None:
    try:
        return datetime.strptime(str(stamp), "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None


def _reaction_summary(reaction: dict | None) -> str:
    if not reaction:
        return "no recorded reaction"
    parts = []
    if reaction.get("message_reported"):
        parts.append("reported")
    if reaction.get("muted_after_message"):
        parts.append("muted the chat afterwards")
    if reaction.get("notification_dismissed"):
        parts.append("dismissed the notification")
    if reaction.get("message_replied"):
        parts.append("replied")
    if reaction.get("message_opened"):
        parts.append("opened")
    else:
        parts.append("never opened")
    minutes = reaction.get("reaction_time_minutes")
    if reaction.get("message_opened") and minutes is not None:
        parts.append(f"reacted in {int(minutes)} min")
    return ", ".join(parts)


@dataclass
class Candidate:
    record: dict
    score: float
    channel_match: bool

    @property
    def message_id(self) -> str:
        return self.record["message_id"]

    def render(self, transcripts: dict[str, str] | None = None) -> str:
        r = self.record
        text = r.get("message_text")
        if not text and r.get("media_id"):
            resolved = (transcripts or {}).get(r["media_id"])
            label = f"[{r.get('media_type')} {r['media_id']}]"
            text = f"{label} {resolved}" if resolved else label
        text = " ".join(str(text or "").split())
        if len(text) > 260:
            text = text[:257] + "..."
        where = r.get("group_id") or r.get("business_id") or r.get("sender_user_id") or "-"
        return (
            f"- {r['message_id']} [{r.get('created_at')}] via {where}\n"
            f"    text: {text}\n"
            f"    user reaction: {_reaction_summary(r.get('reaction'))}"
        )


def gather(message: dict, history: list[dict], limit: int = 8) -> list[Candidate]:
    """Shortlist historical messages worth showing the model."""
    incoming_text = message.get("message_text") or ""
    now = _parse(message.get("created_at"))

    scored: list[Candidate] = []
    for record in history:
        if record["message_id"] == message.get("message_id"):
            continue

        channel_match = False
        score = 0.0
        conv = message.get("conversation_type")
        if conv == "group" and record.get("group_id") == message.get("group_id"):
            channel_match = True
            score += 0.5
            # Same person in the same group is stronger than the group at large.
            if record.get("sender_user_id") == message.get("sender_user_id"):
                score += 0.35
        elif conv == "business" and record.get("business_id") == message.get("business_id"):
            channel_match = True
            score += 0.7
        elif conv == "personal" and record.get("sender_user_id") == message.get(
            "sender_user_id"
        ):
            channel_match = True
            score += 0.7

        sim = similarity(incoming_text, record.get("message_text"))
        score += sim * 1.2

        # A near-duplicate the user ignored is the whole argument for muting,
        # so let strong reactions surface even from a different channel.
        reaction = record.get("reaction") or {}
        if sim > 0.15:
            if reaction.get("message_reported"):
                score += 0.4
            if reaction.get("muted_after_message"):
                score += 0.3
            if reaction.get("notification_dismissed"):
                score += 0.2
            if reaction.get("message_replied"):
                score += 0.15

        if record.get("media_id") and record.get("media_id") == message.get("media_id"):
            score += 0.5

        stamp = _parse(record.get("created_at"))
        if now and stamp:
            days = abs((now - stamp).days)
            score += max(0.0, 0.25 - days / 400.0)

        if score > 0.25:
            scored.append(Candidate(record=record, score=score, channel_match=channel_match))

    scored.sort(key=lambda c: c.score, reverse=True)

    # The history repeats the same spam verbatim many times over. Two copies
    # prove the pattern; ten just crowd out the rest of the evidence, so keep
    # at most two per distinct text and let other candidates through.
    kept: list[Candidate] = []
    seen_text: dict[frozenset[str], int] = {}
    for candidate in scored:
        key = frozenset(tokens(candidate.record.get("message_text")))
        count = seen_text.get(key, 0)
        if key and count >= 2:
            continue
        seen_text[key] = count + 1
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept
