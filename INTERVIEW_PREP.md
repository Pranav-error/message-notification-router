# Interview prep — Message Notification Router

Study material for the AI Judge interview. **Not part of the submission** (excluded from
`code.zip`). The judge asks about the codebase; this is what is in it and why.

---

## Say this first, if asked "what did you build"

> Every incoming WhatsApp message gets routed to notify, digest, or mute — personalised per user.
> The pipeline resolves images and voice notes to text, builds a dossier of that user's
> relationship with that sender, retrieves similar past messages *plus how the user reacted to
> them*, computes deterministic risk signals, and then decides. A two-tier cascade settles about a
> quarter of messages with no model call at all, and a safety floor forces mute on
> credential-harvesting and prompt-injection regardless of what the model says.

Then the one-line thesis: **routing is personal, safety is not.** The same text routes differently
for different people, but a scam is muted for everyone.

---

## The files, and the one decision in each

| file | what it does | the decision to defend |
|---|---|---|
| `data.py` | loads 13 CSVs into indexed lookups | joins each history message to its user reaction once at load, so retrieval never re-joins |
| `signals.py` | deterministic pre-computation | arithmetic (quiet hours, account age, dismissal rates) belongs in code, not in a model prompt |
| `retrieval.py` | ranks the user's history for evidence | ranks on **reaction strength**, not just similarity — a near-duplicate the user muted is the whole argument for muting |
| `context.py` | renders the dossier | sender text is fenced in `<<<MESSAGE_TEXT` delimiters so the model can tell instructions from content |
| `understanding.py` | vision + ASR | resolved **once per media file** and cached; a poster sent to 4 users costs 1 vision call |
| `belief.py` | log-odds fusion of signals | fusion keeps magnitude that booleans discard (4-of-4 ignored ≠ 12-of-12) |
| `cascade.py` | tier-1 settle-or-escalate | settling requires confidence in the **action and the type** |
| `prompts.py` | system prompt + schemas | message-type precedence list; evidence capped at 2 ids |
| `reasons.py` | canonical rationale catalog | `reason` is graded on *consistency*, so it is selected, not free-written |
| `router.py` | decision + post-processing | model proposes, code disposes: safety floor, evidence validation, calibration |
| `llm.py` | provider access | two backends behind one call; lazy client so cached runs need no key |

---

## The questions you will get

**"Why a VLM instead of OCR/Tesseract?"**
The router needs four things from an image; OCR gives one. Verbatim text, *what kind of artifact*
it is, *whether it solicits payment or credentials*, and a description when there is **no text at
all**. That last case is real here — several images are plain photographs. Tesseract returns an
empty string and the message becomes untypeable; the VLM returns "clothing rack in a boutique",
which is what makes it `promotion` rather than `unknown`.

**"Why is ASR local instead of an API?"**
Forced, then justified. The Anthropic API has no audio input. Local Whisper costs nothing per
call, and keeps voice notes on-device — the right default for a messaging product.

**"How do you know the cascade doesn't hurt accuracy?"**
Measured, not assumed. Zero disagreements against the LLM-on-everything baseline across all 110
messages, and identical scores on the labelled set. `--no-cascade` reproduces the baseline so
anyone can diff it. Independently, on 78 held-out behavioural rows, **tier 1 produced 0 errors**
(14/14) while all 19 disagreements came from the model.

**"Why Opus and not a cheaper model?"**
Measured all three: Opus 100% / Sonnet 96.7% / Haiku 93.3% on action. Haiku produced a false
interrupt and a missed message — the two failures the product exists to prevent — and its
confidence when wrong (0.865) is indistinguishable from when right (0.881), so no threshold
recovers it.

**"What's your evidence precision? It's only 0.57."**
Volunteer this before they find it. The gold evidence ids are paired *positionally* with the
sample index (`sample_msg_044` → `message_0049`). That is a dataset-seeding artifact, not a
semantic rule. Matching it would mean keying on an id pattern, which the rules forbid and which
would not generalise to hidden labels. Retrieval instead cites the most relevant same-channel
prior message carrying a real user reaction.

**"How do you handle prompt injection?"**
Three layers. The sender's text is fenced and labelled as untrusted in the dossier; the system
prompt says an attempt to instruct the router is itself evidence of manipulation; and the safety
floor in `router.py` forces mute in post-processing, so a persuasive message cannot argue its way
out. Demonstrated live: a novel injection is muted in 0.00s with no model call.

**"Isn't this too expensive to actually run?" / "How does this scale?"**
This is the question to *want*. Separate two numbers immediately, because judges may conflate
them:

- **Build cost** — about $1.40 total, across every experiment, ablation and model comparison.
  That is development spend on a 110-row dataset. It is not what the system costs to operate and
  should not be read as such.
- **Unit cost at steady state** — $0.0069 per message, which is **$250/user/year** at 100
  messages/day. That is the number that matters, and it is not viable. Say so before they do.

Then give the analysis rather than a defence: 46% of the remaining cost is the dossier, 33% is
output tokens, 21% is the cached prefix — so compressing prompts cannot fix it, because deleting
the dossier entirely still leaves 54%. The cost is in calling a frontier model 76% of the time.

Then the migration path, with the numbers in the README table: a learned tier-1 trained on the
412 behavioural labels already in the repo takes escalation from 76% to ~15% ($49/user/yr), Haiku
on the remainder reaches ~$10, and a distilled on-device model reaches ~$2. Close with the
principle: **the LLM is the prototype's decision engine and the product's fallback** — it should
earn its cost on genuinely novel messages, never on the tenth "good morning" forward from a known
number. The cascade already demonstrates that transition for 24% of traffic, with measured
evidence it costs no accuracy.

Volunteering "my own system is not production-viable, here is the path and here is what it costs
at each stage" reads as engineering judgment. Claiming $0.76 is cheap reads as not having done
the arithmetic.

**"What would you do with more time?"**
Semantic compression of the dossier (measured: only ~11% of the request, so low value), a larger
gold set, and per-user threshold learning. Say what you *measured and rejected*, not a wishlist.

---

## The four stories that show judgment

Lead with these. They are stronger than any metric because they show diagnosis, not tuning.

**0. I tested on data the corpus never contained, and it found a real bug.**
Every other number is on the organizer's synthetic corpus. So I wrote 30 novel messages —
including scam families absent from the dataset (crypto doubling, fake police, romance
advance-fee), injections phrased differently, and a real emergency wearing scam clothing.
28/30 agreement. More importantly it caught a genuine defect: a legitimate OTP *delivery*
("your one-time password is 448120… do not share it") was being muted as scam by tier 1, so the
model never saw it. In production that mutes the code the user is waiting for. The regex could
not tell a credential being delivered from one being requested; the fix walks each transmission
verb, skips negated ones, and requires a credential object after it — so a scam that fakes the
"do not share" warning and then asks anyway is still caught. Zero regressions on the corpus.
Two of my own test labels were also wrong, and I corrected the test rather than the router.

**0b. External validation: 96.7% on a public human-labelled SMS corpus.**
The generalization suite is still me scoring my own system, so I also ran the UCI SMS Spam
Collection — 5,406 real messages labelled by human annotators, downloaded at runtime. 96.7%
agreement, 29/30 spam caught, 29/30 legitimate left alone. Be precise about scope: SMS has no
personalization context, so it tests the content-safety path only, and the domain is 2011 UK SMS
against 2026 Indian WhatsApp.

The honest part, and the bit worth telling: the single false positive was "just send your account
details and the money will be sent to you", muted as scam. My first instinct was that the missing
sender history caused it, so I built the ablation into the script — re-route the same text from a
known contact. **It did not flip:** mute 5/5 either way. The content alone drives it. I had
claimed the opposite after one ad-hoc run and corrected it when the reproducible version
disagreed. If asked whether that behaviour is wrong: it is an error against the label, but a
known contact asking for bank account details is also what a compromised account looks like, so a
safety-first router erring toward mute there is defensible.

**1. Belief fusion broke, and the fix was an invariant.**
Porting the PTT belief graph regressed message_type 96.7% → 93.3%. Cause: banding on the
posterior alone. A high posterior says a message is confidently *unwanted*; it does not say *what
kind* of unwanted. It settled a cold sales voice note as `promotion` when it was `spam`. Fix:
settling requires confidence in the type too, and only content-level fraud evidence may type
something `scam` without the model. Restored to 96.7%.

**2. A preference rule outranked a safety signal.**
The forwarder rule claimed `msg_073`/`msg_074` — phishing messages that happen to be heavily
forwarded — and typed them `forward` instead of `scam`. `msg_074` was an advance-fee property
scam where the user had **reported 5 of that sender's 8 prior messages**. Fix, as a principle:
dismissing is a boredom signal, reporting is a fraud signal; a preference rule must never outrank
a risk signal.

**3. The obvious optimisation was the wrong one.**
Planned to compress the dossier. Profiled first: dossier ~830 tokens, system prompt **2,863** and
identical on every call. Compression would have saved ~11% and invalidated every cached decision.
Prompt caching instead — $0, zero accuracy risk, verified by `cache_read=3709` on the second call.
Cold run $2.80 → $0.76.

---

## Numbers to have ready

| | |
|---|---|
| gold labels (30) | 100% action, 96.7% type, Brier 0.020 |
| behavioural (78 held-out) | 87.2% mute/not-mute; cascade 14/14, model 19 errors |
| ablation `--no-media` | 96.7% — **misses the "Dad is unwell" voice note** |
| escalation rate | 76.4% (26 of 110 never reach a model) |
| cold-run cost | $0.76 |
| models | Opus 100 / Sonnet 96.7 / Haiku 93.3 |

---

## Limitations to volunteer before you are asked

Volunteering these reads as confidence; being caught by them reads as overclaiming.

- **Only 30 gold labels.** One message is 3.3 points. Say "100% action on the labelled sample",
  never "100% accurate".
- **Evidence precision ~0.57**, for the positional-artifact reason above.
- **The behavioural eval uses proxy labels.** A user opening a phishing message does not make
  muting it wrong — which is why those numbers were deliberately **not** used to tune the pipeline.
  Optimising toward them would train the router to stop muting the messages users fall for.
- **Model comparison is on 30 rows**, not the full set.

---

## If asked how you used AI to build this

Answer straight: AI-assisted throughout, which the rules explicitly permit. What is yours is the
architecture — the cascade and belief-fusion patterns are ported from your own earlier
INFRRD mortgage pipeline (`DocCompiler + PTT`), and the design decisions above are ones you can
defend on their merits. Do not overclaim line-by-line authorship; the judge has the transcript.
