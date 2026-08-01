"""Deterministic signals computed before the model sees anything.

Two reasons these live outside the prompt:

1. Some of them are arithmetic (dismissal rates, account age, quiet-hour
   overlap). A language model should not be asked to do arithmetic it can be
   handed for free.
2. The credential-harvesting checks act as a floor. The problem statement says
   clear scam or safety risk is muted *regardless* of the user's usual
   engagement, so that decision must not be something a persuasive message can
   talk its way out of.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any

# Credential/verification harvesting. These are about the *ask*, not the topic:
# a bank talking about your statement is fine, a bank asking you to type an OTP
# into a link is not.
CREDENTIAL_ASK = re.compile(
    r"\b(otp|one[- ]time (?:code|password)|cvv|pin\b|password|login code|"
    r"verification code|6[- ]digit)\b",
    re.I,
)
ACCOUNT_PRESSURE = re.compile(
    r"\b(account|profile|access|wallet|kyc|card)\b[^.]{0,60}\b"
    r"(block|blocked|suspend|suspended|restrict|restricted|expire|expires|"
    r"expiring|deactivat|lock|locked|close|closing)\w*",
    re.I,
)
URGENT_DEADLINE = re.compile(
    r"\b(immediately|right now|within \d+ ?(min|hour)|before midnight|"
    r"in \d+ ?(minutes|hours)|final (notice|warning|reminder)|last chance|"
    r"today only|before (the )?(day|deadline) ends)\b",
    re.I,
)
PAYMENT_ASK = re.compile(
    r"\b(pay|payment|fee|transfer|upi|deposit|recharge|reattempt|release|"
    r"processing charge|refund)\b",
    re.I,
)
SHORTENER = re.compile(
    r"\b(bit\.ly|tinyurl|t\.co|goo\.gl|rb\.gy|cutt\.ly|is\.gd|shorturl|"
    r"wame\.pro|weurl\.co)\b",
    re.I,
)
LINK = re.compile(r"(https?://\S+|\b[a-z0-9-]+\.(?:com|in|net|org|co|io|me|pro|sg|bank)\b)", re.I)

# Attempts to talk to the router rather than to the user.
INJECTION = re.compile(
    r"\b(ignore (all |any )?(previous|prior|above) (instruction|rule|routing)|"
    r"disregard (the )?(previous|prior|above)|system prompt|"
    r"mark this (message )?as|treat this as (high priority|urgent|notify)|"
    r"you are an? (ai|assistant|router)|override .{0,20}(rule|setting))\b",
    re.I,
)

CHAIN_FORWARD = re.compile(
    r"\b(forward(ed)? (this |as received|to )|share (this )?with \d+|"
    r"share in (family |all )?groups?|pls forward|before (sunset|night)|"
    r"send to everyone)\b",
    re.I,
)


@dataclass
class Signals:
    """Everything computed deterministically about one incoming message."""

    facts: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    impostor_score: float = 0.0
    credential_harvest: bool = False
    injection_attempt: bool = False
    in_quiet_hours: bool = False

    def as_lines(self) -> list[str]:
        return [f"{k}: {v}" for k, v in self.facts.items() if v is not None]


def _parse_window(window: str | None) -> tuple[time, time] | None:
    if not window or "-" not in str(window):
        return None
    start, end = str(window).split("-", 1)
    try:
        sh, sm = (int(x) for x in start.strip().split(":"))
        eh, em = (int(x) for x in end.strip().split(":"))
    except ValueError:
        return None
    return time(sh, sm), time(eh, em)


def in_quiet_hours(created_at: str | None, window: str | None) -> bool:
    parsed = _parse_window(window)
    if not parsed or not created_at:
        return False
    try:
        stamp = datetime.strptime(str(created_at), "%Y-%m-%d %H:%M")
    except ValueError:
        return False
    start, end = parsed
    now = stamp.time()
    if start <= end:
        return start <= now <= end
    # Window wraps past midnight, e.g. 22:00-07:00.
    return now >= start or now <= end


def impostor_risk(business: dict | None) -> tuple[float, list[str]]:
    """How much a business sender looks like a brand impersonator.

    Deliberately graded rather than boolean. A long-lived verified brand that
    sends through a link shortener is a marketing choice; a three-week-old
    unverified account on a lookalike domain with 60 reports is not.
    """
    if not business:
        return 0.0, []

    flags: list[str] = []
    score = 0.0

    official = (business.get("official_domain") or "").strip().lower()
    used = (business.get("domain_used_by_sender") or "").strip().lower()
    verified = int(business.get("verified") or 0)
    age = float(business.get("account_age_days") or 0)
    reports = float(business.get("user_reports_30d") or 0)
    sent = float(business.get("messages_sent_30d") or 0)

    if official and used and official != used:
        # A lookalike domain reuses the brand token: hdfc.bank.in -> hdfcbank-kyc.in
        brand_token = re.split(r"[.-]", official)[0]
        if brand_token and brand_token in used:
            score += 0.35
            flags.append(f"sender domain '{used}' imitates official '{official}'")
        else:
            score += 0.15
            flags.append(f"sender domain '{used}' differs from official '{official}'")

    if not verified:
        score += 0.2
        flags.append("business account is not verified")

    if age and age < 60:
        score += 0.25
        flags.append(f"account is only {int(age)} days old")

    report_rate = reports / max(sent, 1.0)
    if reports >= 25:
        score += 0.25
        flags.append(f"{int(reports)} user reports in 30 days")
    elif report_rate > 0.02 and reports >= 10:
        score += 0.1
        flags.append(f"elevated report rate ({report_rate:.1%})")

    return min(score, 1.0), flags


def compute(message: dict, ctx: dict) -> Signals:
    """Build the signal bundle for one incoming message.

    ``ctx`` carries the already-resolved user / group / membership / business
    records so this function stays free of dataset lookups.
    """
    text = message.get("message_text") or ""
    transcript = ctx.get("transcript") or ""
    # Voice notes carry their payload in the transcript, so scan both.
    scannable = f"{text}\n{transcript}".strip()

    sig = Signals()
    user = ctx.get("user") or {}
    business = ctx.get("business")

    sig.in_quiet_hours = in_quiet_hours(
        message.get("created_at"), user.get("do_not_disturb_window")
    )
    sig.impostor_score, brand_flags = impostor_risk(business)
    sig.risk_flags.extend(brand_flags)

    asks_credentials = bool(CREDENTIAL_ASK.search(scannable))
    pressures_account = bool(ACCOUNT_PRESSURE.search(scannable))
    has_deadline = bool(URGENT_DEADLINE.search(scannable))
    has_shortener = bool(SHORTENER.search(scannable))
    asks_payment = bool(PAYMENT_ASK.search(scannable))
    has_link = bool(LINK.search(scannable))

    # The hard floor: a request for a secret, combined with either manufactured
    # urgency about account status or an off-platform link to type it into.
    sig.credential_harvest = asks_credentials and (
        pressures_account or has_deadline or has_link or has_shortener
    )
    if sig.credential_harvest:
        sig.risk_flags.append("asks for OTP/password alongside account pressure or a link")
    elif asks_credentials:
        sig.risk_flags.append("mentions OTP or login credentials")

    if pressures_account and has_deadline:
        sig.risk_flags.append("threatens account block within a deadline")
    if has_shortener:
        sig.risk_flags.append("uses a link shortener or non-brand redirect domain")
    if asks_payment and pressures_account:
        sig.risk_flags.append("links payment to restoring account access")

    sig.injection_attempt = bool(INJECTION.search(scannable))
    if sig.injection_attempt:
        sig.risk_flags.append("message text tries to instruct the routing system")

    forwarded = int(message.get("forwarded_count") or 0)
    chain = bool(CHAIN_FORWARD.search(scannable))
    if forwarded >= 5:
        sig.risk_flags.append(f"forwarded {forwarded} times (chain-forward pattern)")

    sig.facts = {
        "sent_during_quiet_hours": f"yes ({user.get('do_not_disturb_window')})"
        if sig.in_quiet_hours
        else "no",
        "forwarded_count": forwarded,
        "chain_forward_language": "yes" if chain else "no",
        "asks_for_credentials": "yes" if asks_credentials else "no",
        "account_block_pressure": "yes" if pressures_account else "no",
        "artificial_deadline": "yes" if has_deadline else "no",
        "contains_link": "yes" if has_link else "no",
        "brand_impostor_score": f"{sig.impostor_score:.2f}" if business else None,
    }
    return sig
