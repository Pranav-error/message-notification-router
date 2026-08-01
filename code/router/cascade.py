"""Confidence-thresholded model cascade.

Most incoming messages are not judgment calls. A three-week-old lookalike bank
account asking for an OTP is a scam whoever receives it; the eleventh forwarded
blessing from a sender whose last ten forwards the user silently dismissed is
noise. Sending those to a frontier model buys nothing but latency and spend.

So routing runs in two tiers, following the FrugalGPT cascade pattern
(Chen et al. 2023, arXiv:2305.05176):

    Tier 1  belief fusion over the computed signals and the user's own
            reaction history (see `belief.py`). Free, instant, auditable.
    Tier 2  the language model, for everything Tier 1 will not claim.

Tier 1 settles only when the fused posterior clears 0.90, and only ever toward
`mute` — see `belief.py` for why the low band deliberately escalates instead of
settling. A cascade that guesses to save money is a worse product, not a cheaper
one, so the escalation rate and the spend avoided are recorded as measured
numbers rather than claims.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from . import belief, context

# Anthropic list prices, USD per token, for the spend-avoided estimate.
PRICE_PER_INPUT_TOKEN = 5.00 / 1_000_000
PRICE_PER_OUTPUT_TOKEN = 25.00 / 1_000_000
# Measured on the full 110-message run: 517,740 in / 12,950 out over 110 calls.
AVG_INPUT_TOKENS_PER_CALL = 4707
AVG_OUTPUT_TOKENS_PER_CALL = 118


@dataclass
class Verdict:
    """A Tier-1 decision, or None-valued when the rule declines to claim it."""

    action: str
    message_type: str
    rationale: str
    confidence: float
    rule: str
    evidence: list[str] = field(default_factory=list)
    posterior: float = 0.0
    explanation: str = ""


@dataclass
class CascadeStats:
    """Escalation and cost accounting across a run."""

    resolved: int = 0
    escalated: int = 0
    by_rule: dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, verdict: Verdict | None) -> None:
        with self._lock:
            if verdict is None:
                self.escalated += 1
            else:
                self.resolved += 1
                self.by_rule[verdict.rule] = self.by_rule.get(verdict.rule, 0) + 1

    @property
    def total(self) -> int:
        return self.resolved + self.escalated

    def summary(self) -> dict[str, Any]:
        total = self.total or 1
        avoided = self.resolved * (
            AVG_INPUT_TOKENS_PER_CALL * PRICE_PER_INPUT_TOKEN
            + AVG_OUTPUT_TOKENS_PER_CALL * PRICE_PER_OUTPUT_TOKEN
        )
        return {
            "messages": self.total,
            "resolved_by_rules": self.resolved,
            "escalated_to_model": self.escalated,
            "escalation_rate": round(self.escalated / total, 3),
            "by_rule": dict(sorted(self.by_rule.items(), key=lambda kv: -kv[1])),
            "estimated_spend_avoided_usd": round(avoided, 4),
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            "cascade — escalation report",
            f"  messages routed        {s['messages']}",
            f"  resolved by rules      {s['resolved_by_rules']} "
            f"({100 * s['resolved_by_rules'] / (s['messages'] or 1):.1f}%)",
            f"  escalated to model     {s['escalated_to_model']} "
            f"({100 * s['escalation_rate']:.1f}%)",
            f"  spend avoided (est.)   ${s['estimated_spend_avoided_usd']:.4f}",
        ]
        if s["by_rule"]:
            lines.append("  rules fired:")
            lines.extend(f"    {name:24s} {count}" for name, count in s["by_rule"].items())
        return "\n".join(lines)


# ----------------------------------------------------------------- evidence


def _sender_reaction_profile(dataset, ctx: context.MessageContext) -> dict[str, int]:
    """How this user has historically reacted to this exact sender."""
    sender = ctx.message.get("sender_user_id")
    if not sender:
        return {}
    prior = [
        r
        for r in dataset.history_by_user.get(ctx.message["user_id"], [])
        if r.get("sender_user_id") == sender
    ]
    if not prior:
        return {}
    reactions = [r.get("reaction") or {} for r in prior]
    return {
        "count": len(prior),
        "opened": sum(1 for x in reactions if x.get("message_opened")),
        "replied": sum(1 for x in reactions if x.get("message_replied")),
        "dismissed": sum(1 for x in reactions if x.get("notification_dismissed")),
        "reported": sum(1 for x in reactions if x.get("message_reported")),
    }


def _negative_evidence(ctx: context.MessageContext, limit: int = 2) -> list[str]:
    """Historical ids where the user visibly rejected a similar message."""
    scored = []
    for candidate in ctx.candidates:
        reaction = candidate.record.get("reaction") or {}
        weight = (
            3 * int(bool(reaction.get("message_reported")))
            + 2 * int(bool(reaction.get("muted_after_message")))
            + int(bool(reaction.get("notification_dismissed")))
            + int(not reaction.get("message_opened"))
        )
        if weight:
            scored.append((weight, candidate.score, candidate.message_id))
    scored.sort(reverse=True)
    return [mid for _, _, mid in scored[:limit]]


# Content- or identity-level fraud evidence. These type a message as `scam` on
# their own; weaker circumstantial evidence (a reported sender, a merely
# suspicious domain) does not, because the same facts fit an unwanted-but-legal
# promotion just as well.
DECISIVE_SCAM = ("prompt_injection", "credential_harvest", "brand_impostor")
RISK_SIGNALS = DECISIVE_SCAM + ("sender_reported", "suspicious_sender")


def _type_and_rationale(names: set[str], ctx) -> tuple[str, str] | None:
    """Choose message_type and rationale, or None if the type is a judgment call.

    A high posterior says the message is confidently unwanted; it does not say
    *what kind* of unwanted. Settling requires confidence in both, so this
    returns None — escalating — whenever the type is genuinely arguable. That
    distinction is what the first fused version got wrong: it settled
    `opted_out_of_promotions` as `promotion` on a cold sales voice note that
    was really `spam`, and on an unverified lookalike payments account that was
    really `scam`.
    """
    if "prompt_injection" in names:
        return "scam", "prompt_injection"
    if "credential_harvest" in names:
        return "scam", "otp_phishing"
    if "brand_impostor" in names:
        return "scam", "brand_impersonation"

    # Below here nothing is fraud-shaped at the content level. Any residual
    # risk signal makes scam-vs-unwanted a judgment call: escalate.
    if names & set(RISK_SIGNALS):
        return None

    if names & {"sender_always_ignored", "chain_forward", "heavily_forwarded"}:
        text = f"{ctx.message.get('message_text') or ''} {ctx.transcript or ''}".lower()
        greeting_words = (
            "good morning", "good night", "blessing", "smile", "positive", "stay happy",
        )
        if any(word in text for word in greeting_words):
            return "greeting", "repeat_forwarder"
        return "forward", "repeat_forwarder"

    if names & {"opted_out_of_promotions", "dismisses_this_business"}:
        business = ctx.business
        sig = ctx.signals
        # `promotion` vs `spam` turns on whether there is a real brand behind
        # the sender. Only an established, verified, clean account is safely
        # typeable without the model.
        if (
            business
            and int(business.get("verified") or 0)
            and float(business.get("account_age_days") or 0) >= 365
            and sig
            and sig.impostor_score <= 0.2
            and not sig.risk_flags
        ):
            return "promotion", "opted_out_marketing"
        return None

    return None


def decide(dataset, ctx: context.MessageContext) -> Verdict | None:
    """Return a Tier-1 verdict, or None to escalate to the model.

    Fuses the independent signals into one posterior and settles only when that
    posterior clears the high band. Everything else is a judgment call.
    """
    signals = belief.gather_signals(dataset, ctx)
    if not signals:
        return None

    fused = belief.fuse(signals)
    if fused.posterior < belief.SETTLE_HIGH:
        return None

    names = set(fused.names)
    typed = _type_and_rationale(names, ctx)
    if typed is None:
        # Confidently unwanted, but the *type* is arguable — escalate.
        return None
    message_type, rationale = typed

    # Confidence reports the fused posterior, clamped into the mute band so it
    # stays comparable with the model-produced rows.
    confidence = round(min(max(fused.posterior, 0.79), 0.91), 2)
    return Verdict(
        action="mute",
        message_type=message_type,
        rationale=rationale,
        confidence=confidence,
        rule=fused.decisive,
        evidence=_negative_evidence(ctx, 2),
        posterior=fused.posterior,
        explanation=fused.explain(),
    )
