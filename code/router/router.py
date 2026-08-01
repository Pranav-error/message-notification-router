"""The router: dossier in, routing decision out.

The model proposes; this module disposes. Post-processing does three jobs the
model should not be trusted with alone:

- a safety floor, so a credential-harvesting or router-manipulating message
  cannot be argued up out of `mute`
- evidence validation, so only ids that were actually offered as candidates can
  be cited
- confidence calibration, so the numbers land in a consistent, comparable band
  instead of drifting per model
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from . import cascade, context, ensemble, reasons
from .data import Dataset
from .llm import Client
from .prompts import RESPONSE_SCHEMA, routing_messages
from .understanding import MediaIndex

# Ground-truth confidences in the worked examples sit in a narrow band that
# differs by action. Anchoring to it keeps calibration comparable across models
# and stops one model's habitual 0.95 from reading as more certain than
# another's habitual 0.7.
CONFIDENCE_BANDS = {
    "notify": (0.83, 0.92),
    "digest": (0.76, 0.86),
    "mute": (0.79, 0.90),
}


@dataclass
class Decision:
    message_id: str
    action: str
    message_type: str
    reason: str
    confidence: float
    evidence_message_ids: str
    rationale: str = ""
    risk_note: str = ""
    overrides: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_output_row(self) -> dict[str, Any]:
        """Exactly the six submission columns, in order."""
        return {
            "message_id": self.message_id,
            "action": self.action,
            "message_type": self.message_type,
            "reason": self.reason,
            "confidence": self.confidence,
            "evidence_message_ids": self.evidence_message_ids,
        }


def calibrate(action: str, raw_confidence: Any) -> float:
    try:
        value = float(raw_confidence)
    except (TypeError, ValueError):
        value = 0.7
    value = min(max(value, 0.0), 1.0)
    low, high = CONFIDENCE_BANDS.get(action, (0.76, 0.90))
    # Model confidences cluster above 0.5; stretch that range across the band.
    normalized = min(max((value - 0.5) / 0.5, 0.0), 1.0)
    return round(low + (high - low) * normalized, 2)


def _apply_safety_floor(
    result: dict[str, Any], ctx: context.MessageContext
) -> tuple[dict[str, Any], list[str]]:
    """Force `mute` for the cases the spec says must never interrupt."""
    overrides: list[str] = []
    sig = ctx.signals
    if not sig:
        return result, overrides

    if sig.injection_attempt and result.get("action") != "mute":
        result = dict(result, action="mute", message_type="scam", rationale="prompt_injection")
        overrides.append("prompt_injection")
    elif sig.credential_harvest and result.get("action") != "mute":
        result = dict(result, action="mute")
        if result.get("message_type") not in {"scam", "spam"}:
            result["message_type"] = "scam"
        if reasons.REASONS.get(result.get("rationale", ""), {}).get("action") != "mute":
            result["rationale"] = "otp_phishing"
        overrides.append("credential_harvest")

    # An unverified, newly created, heavily reported account wearing a known
    # brand's name is impersonation regardless of how helpful the copy reads.
    if sig.impostor_score >= 0.75 and result.get("action") != "mute":
        result = dict(result, action="mute")
        if result.get("message_type") not in {"scam", "spam"}:
            result["message_type"] = "scam"
        result["rationale"] = "brand_impersonation"
        overrides.append("brand_impersonation")

    return result, overrides


def _clean_evidence(raw: Any, ctx: context.MessageContext) -> str:
    allowed = set(ctx.candidate_ids)
    if not isinstance(raw, list):
        raw = []
    kept = [str(x).strip() for x in raw if str(x).strip() in allowed]
    # Preserve the model's ordering but drop duplicates.
    seen, ordered = set(), []
    for mid in kept:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    return ";".join(ordered[:2]) if ordered else "none"


def _reconcile_rationale(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the rendered reason consistent with the final action."""
    rationale = result.get("rationale", "")
    entry = reasons.REASONS.get(rationale)
    if entry and entry["action"] != result.get("action"):
        # The floor moved the action; pick a rationale that agrees with it.
        fallback = {
            "mute": "ignored_similar_history",
            "digest": "useful_not_urgent",
            "notify": "direct_request",
        }.get(result["action"], rationale)
        result = dict(result, rationale=fallback)
    return result


def route_one(
    dataset: Dataset,
    message: dict,
    client: Client,
    media_index: MediaIndex | None = None,
    stats: cascade.CascadeStats | None = None,
    use_cascade: bool = True,
    use_ensemble: bool = False,
) -> Decision:
    ctx = context.build(dataset, message)

    if media_index and message.get("media_id"):
        resolved = media_index.resolve(
            message.get("media_id"),
            message.get("media_type"),
            dataset.media_path(message.get("media_id")),
        )
        ctx.image_caption = resolved["caption"]
        ctx.transcript = resolved["transcript"]
        # Voice transcripts carry scam text, so recompute signals over them.
        if ctx.transcript:
            ctx = context.build(dataset, message, transcripts={message["media_id"]: ctx.transcript})
            ctx.image_caption = resolved["caption"]

    # Tier 1: settle the unambiguous cases without spending a model call.
    verdict = cascade.decide(dataset, ctx) if use_cascade else None
    if stats is not None:
        stats.record(verdict)
    if verdict is not None:
        return Decision(
            message_id=message["message_id"],
            action=verdict.action,
            message_type=verdict.message_type,
            reason=reasons.render(verdict.rationale, verdict.action),
            confidence=verdict.confidence,
            evidence_message_ids=_clean_evidence(verdict.evidence, ctx),
            rationale=verdict.rationale,
            risk_note=f"resolved by cascade rule: {verdict.rule}",
            overrides=[f"cascade:{verdict.rule}"],
            raw={"tier": 1, "rule": verdict.rule},
        )

    # Tier 2: a genuine judgment call — hand it to the model.
    dossier = context.render(dataset, ctx)
    result = client.json(
        messages=routing_messages(dossier),
        schema=RESPONSE_SCHEMA,
        schema_name="routing_decision",
    )

    # Optional second opinion on the messages tier 1 declined to settle.
    if use_ensemble:
        second = ensemble.review(client, dossier)
        if (second.get("action"), second.get("message_type")) != (
            result.get("action"), result.get("message_type")
        ):
            result = ensemble.adjudicate(client, dossier, result, second)
            result["_adjudicated"] = True

    result, overrides = _apply_safety_floor(result, ctx)
    result = _reconcile_rationale(result)

    action = result.get("action", "digest")
    confidence = calibrate(action, result.get("confidence"))
    if overrides:
        # A deterministic rule fired; that is the most certain we ever are.
        confidence = max(confidence, CONFIDENCE_BANDS["mute"][1] - 0.02)

    return Decision(
        message_id=message["message_id"],
        action=action,
        message_type=result.get("message_type", "unknown"),
        reason=reasons.render(result.get("rationale", ""), action),
        confidence=confidence,
        evidence_message_ids=_clean_evidence(result.get("evidence_message_ids"), ctx),
        rationale=result.get("rationale", ""),
        risk_note=result.get("risk_note", ""),
        overrides=overrides + (["adjudicated"] if result.get("_adjudicated") else []),
        raw=result,
    )


def route_all(
    dataset: Dataset,
    messages: list[dict],
    client: Client,
    media_index: MediaIndex | None = None,
    workers: int = 6,
    on_done: Any = None,
    stats: cascade.CascadeStats | None = None,
    use_cascade: bool = True,
    use_ensemble: bool = False,
) -> list[Decision]:
    """Route every message, preserving input order."""
    # Resolve media first and single-threaded: several messages share one media
    # file, and this way each file is understood exactly once.
    if media_index:
        for message in messages:
            if message.get("media_id"):
                media_index.resolve(
                    message["media_id"],
                    message.get("media_type"),
                    dataset.media_path(message["media_id"]),
                )

    def work(item: tuple[int, dict]) -> tuple[int, Decision]:
        index, message = item
        decision = route_one(
            dataset, message, client, media_index, stats=stats,
            use_cascade=use_cascade, use_ensemble=use_ensemble,
        )
        if on_done:
            on_done(decision)
        return index, decision

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, enumerate(messages)))

    results.sort(key=lambda pair: pair[0])
    return [decision for _, decision in results]


def to_records(decisions: list[Decision]) -> list[dict]:
    return [d.as_output_row() for d in decisions]


def debug_records(decisions: list[Decision]) -> list[dict]:
    return [asdict(d) for d in decisions]
