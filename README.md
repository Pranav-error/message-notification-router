# Message Notification Router

**HackerRank Orchestrate (August 2026) — Problem Statement: Message Notification Router**

Routes every incoming WhatsApp message to **notify**, **digest**, or **mute** for a specific user,
reasoning over text, image posters/screenshots, and voice notes.

> **Routing is personal, safety is not.** The same text routes differently for different people —
> the dataset contains two identical resale messages labelled `digest` and `mute` — but a
> credential-harvesting scam is muted for everyone, regardless of how much that user engages with
> the sender.

| | |
|---|---|
| action accuracy (30 gold labels) | **100%** |
| message_type accuracy | **96.7%** |
| behavioural agreement (78 held-out messages) | **87.2%** mute/not-mute |
| messages resolved with no model call | **23.6%** |
| cold-run cost | **$0.76**, down from $2.80 |

## Architecture

```
message (text | image | voice)
   -> media understanding      vision model + local Whisper, cached per file
   -> personalization dossier  this user x this sender: history, quiet hours, opt-outs
   -> evidence retrieval       ranked by channel + similarity + reaction strength
   -> deterministic signals    quiet hours, impostor score, credential/injection detection
   -> TIER 1  belief fusion    23.6% settled with no model call
   -> TIER 2  language model   76.4%
   -> post-processing          safety floor, evidence validation, calibration
```

Two patterns are ported from an earlier project of mine, the INFRRD Ideathon mortgage-document
pipeline (`DocCompiler + PTT`): **compile-then-reason** and a **confidence-thresholded cascade**
with **probabilistic belief fusion**.

## Read next

**[`code/README.md`](code/README.md)** — the full document: architecture, the cascade and belief
fusion, evaluation (gold labels, behavioural evaluation, ablations, model comparison), cost
engineering with before/after token accounting, production economics, and the training roadmap.

## Run

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=..." > .env

python code/main.py                      # dataset/messages.csv -> output.csv
python code/evaluation/main.py           # score the labelled samples
python code/demo.py --user u_005 --group group_005 --sender u_050   # live routing
```

The `dataset/` corpus is organizer-provided and excluded from this repository; obtain it from
[the starter repo](https://github.com/interviewstreet/hackerrank-orchestrate-august26).
`CHALLENGE_README.md` and `problem_statement.md` are the organizers' original briefs, kept for
reference.
