"""Belief fusion over independent routing signals.

The first cascade was five boolean rules that either fired or did not. That
works, but it throws away magnitude: a sender the user dismissed 4 times out of
4 and a sender they dismissed 12 times out of 12 produce the same verdict, and
two weak signals pointing the same way produce nothing at all.

This module replaces that with the pattern from the PTT belief graph in the
mortgage pipeline: score several *independent* signals, fuse them into one
posterior, then band the posterior.

    P >= 0.90   confidently unwanted or unsafe: mute here, no model call
    P <  0.90   escalate to the model

Fusion is a log-odds sum (naive Bayes with independence assumed), the same
fusion the PTT graph uses over its five table-continuation signals. The
independence assumption is imperfect — chain-forward language and a high forward
count correlate — so the evidence weights are kept deliberately modest.

Note the deliberate asymmetry: only the *high* band settles. Every signal here
is evidence that a message is unwanted or unsafe, so a high posterior is
genuine evidence for `mute`. A *low* posterior means only "nothing looks wrong",
which does not say whether the message is a same-day school change worth
interrupting for or idle chat that can wait — that is exactly the judgment the
model is there to make. So a low score escalates rather than settling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Posterior thresholds. Deliberately asymmetric: the high band must be very
# certain because it suppresses a message without a model ever seeing it.
SETTLE_HIGH = 0.90

PRIOR = 0.35  # base rate of unwanted/unsafe traffic; tuned on the labelled set


@dataclass
class Signal:
    """One piece of evidence, with the log-odds it contributes."""

    name: str
    weight: float
    detail: str = ""


@dataclass
class Belief:
    posterior: float
    signals: list[Signal] = field(default_factory=list)
    decisive: str = ""

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.signals]

    def explain(self) -> str:
        parts = [f"{s.name}(+{s.weight:.1f})" for s in self.signals]
        return f"P={self.posterior:.2f} <- " + ", ".join(parts) if parts else "P=prior"


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def fuse(signals: list[Signal]) -> Belief:
    """Combine independent evidence into a single posterior."""
    total = _logit(PRIOR) + sum(s.weight for s in signals)
    posterior = _sigmoid(total)
    decisive = max(signals, key=lambda s: s.weight).name if signals else ""
    return Belief(posterior=posterior, signals=signals, decisive=decisive)


def gather_signals(dataset, ctx) -> list[Signal]:
    """Score every independent indicator that this message is unwanted/unsafe.

    Weights are log-odds contributions. A weight of 4.0 is close to decisive on
    its own; 1.0 is a nudge that only matters alongside other evidence.
    """
    signals: list[Signal] = []
    sig = ctx.signals
    message = ctx.message
    if sig is None:
        return signals

    # --- safety evidence -------------------------------------------------
    if sig.credential_harvest:
        signals.append(
            Signal("credential_harvest", 5.0, "asks for a secret under pressure")
        )
    if sig.injection_attempt:
        signals.append(Signal("prompt_injection", 5.0, "tries to instruct the router"))

    if sig.impostor_score >= 0.85:
        signals.append(
            Signal("brand_impostor", 4.0, f"impostor score {sig.impostor_score:.2f}")
        )
    elif sig.impostor_score >= 0.5:
        signals.append(
            Signal("suspicious_sender", 1.5, f"impostor score {sig.impostor_score:.2f}")
        )

    # --- engagement evidence ---------------------------------------------
    profile = _sender_profile(dataset, ctx)
    if profile and profile["count"] >= 3:
        never_engaged = profile["opened"] == 0 and profile["replied"] == 0
        if never_engaged and profile["dismissed"] >= profile["count"]:
            # Scale with sample size: 3 ignored messages is suggestive, 10 is not.
            weight = min(1.4 + 0.35 * (profile["count"] - 3), 3.2)
            signals.append(
                Signal(
                    "sender_always_ignored",
                    weight,
                    f"{profile['dismissed']}/{profile['count']} dismissed, none opened",
                )
            )
        if profile["reported"] > 0:
            # Reporting is a fraud signal, not a boredom signal.
            signals.append(
                Signal("sender_reported", 2.5, f"{profile['reported']} reported")
            )

    forwarded = int(message.get("forwarded_count") or 0)
    if forwarded >= 8:
        signals.append(Signal("chain_forward", 1.6, f"forwarded {forwarded}x"))
    elif forwarded >= 5:
        signals.append(Signal("heavily_forwarded", 1.0, f"forwarded {forwarded}x"))

    # --- business relationship evidence ----------------------------------
    relationship = ctx.relationship
    business = ctx.business
    if relationship and business:
        opted_out = bool(relationship.get("promotions_opted_out_at")) and not int(
            relationship.get("allows_promotions") or 0
        )
        dismissed = float(relationship.get("messages_dismissed_30d") or 0)
        opened = float(relationship.get("messages_opened_30d") or 0)
        if opted_out:
            signals.append(Signal("opted_out_of_promotions", 2.2, "user opted out"))
        if dismissed >= 3 and dismissed > opened:
            signals.append(
                Signal(
                    "dismisses_this_business",
                    1.4,
                    f"{int(dismissed)} dismissed vs {int(opened)} opened",
                )
            )

    # --- near-duplicate the user already rejected -------------------------
    rejected = _rejected_near_duplicate(ctx)
    if rejected:
        signals.append(Signal("rejected_near_duplicate", 2.0, rejected))

    return signals


def _sender_profile(dataset, ctx) -> dict | None:
    sender = ctx.message.get("sender_user_id")
    if not sender:
        return None
    prior = [
        r
        for r in dataset.history_by_user.get(ctx.message["user_id"], [])
        if r.get("sender_user_id") == sender
    ]
    if not prior:
        return None
    reactions = [r.get("reaction") or {} for r in prior]
    return {
        "count": len(prior),
        "opened": sum(1 for x in reactions if x.get("message_opened")),
        "replied": sum(1 for x in reactions if x.get("message_replied")),
        "dismissed": sum(1 for x in reactions if x.get("notification_dismissed")),
        "reported": sum(1 for x in reactions if x.get("message_reported")),
    }


def _rejected_near_duplicate(ctx) -> str:
    """A closely-matching past message the user muted or reported."""
    from .retrieval import similarity

    incoming = ctx.message.get("message_text") or ""
    if not incoming:
        return ""
    for candidate in ctx.candidates:
        reaction = candidate.record.get("reaction") or {}
        if not (reaction.get("muted_after_message") or reaction.get("message_reported")):
            continue
        if similarity(incoming, candidate.record.get("message_text")) >= 0.55:
            return f"{candidate.message_id} was muted/reported by this user"
    return ""
