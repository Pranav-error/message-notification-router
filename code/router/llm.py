"""Model access layer.

Every model call in this project goes through here, behind one `Client.json()`
method that returns schema-validated JSON. Two backends sit behind it:

- **Anthropic** (default) for any bare model id, e.g. ``claude-opus-5``
- **Vercel AI Gateway** for any ``provider/model`` id, e.g. ``openai/gpt-5.2``

That split is what makes the model-comparison sweep in the evaluation harness a
string change rather than a code change, and it means the pipeline is not
stranded if one provider's billing is unavailable.

Requests are cached on disk keyed by (model, messages, schema, params) so
re-running the pipeline or the evals does not re-pay for identical calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"

DEFAULT_MAX_TOKENS = 16000
DEFAULT_EFFORT = "medium"


@dataclass
class Usage:
    """Token/latency accounting, aggregated across a run."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        seconds: float,
        cache_write_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.calls += 1
            self.seconds += seconds
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cache_write_tokens += cache_write_tokens
            self.cache_read_tokens += cache_read_tokens

    def hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "seconds": round(self.seconds, 1),
        }


def _strip_unsupported(schema: Any) -> Any:
    """Remove JSON-Schema keywords the structured-output compiler rejects.

    Array length bounds and numeric ranges are not supported; we enforce those
    in post-processing anyway, so dropping them here is lossless.
    """
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported(v)
            for k, v in schema.items()
            if k not in {"maxItems", "minItems", "minimum", "maximum", "minLength", "maxLength"}
        }
    if isinstance(schema, list):
        return [_strip_unsupported(v) for v in schema]
    return schema


class Client:
    """Schema-constrained model calls with an on-disk cache."""

    def __init__(
        self,
        model: str,
        cache_dir: str | Path = ".cache/llm",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        max_retries: int = 5,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_retries = max_retries
        # A "provider/model" id means route through the gateway; a bare id is
        # a first-party Anthropic model.
        self.backend = "gateway" if "/" in model else "anthropic"
        # Flipped to False the first time the API rejects the effort parameter.
        self._supports_effort = True

        # The provider client is built on first use, not here: a fully cached
        # run (re-scoring, ablations, regenerating output.csv) needs no
        # credentials at all, and demanding a key up front would break it.
        self._timeout = timeout
        self._client = None
        self._client_lock = threading.Lock()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()

    def _backend_client(self):
        """Build the provider client on first real call."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            if self.backend == "gateway":
                from openai import OpenAI

                api_key = os.getenv("AI_GATEWAY_API_KEY")
                if not api_key:
                    raise RuntimeError(
                        f"model {self.model!r} routes through the Vercel AI Gateway, "
                        "but AI_GATEWAY_API_KEY is not set."
                    )
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=GATEWAY_BASE_URL,
                    timeout=self._timeout,
                    max_retries=0,
                )
            else:
                import anthropic

                if not os.getenv("ANTHROPIC_API_KEY"):
                    raise RuntimeError(
                        f"model {self.model!r} uses the Anthropic API, but "
                        "ANTHROPIC_API_KEY is not set. Put it in .env or export it."
                    )
                self._client = anthropic.Anthropic(timeout=self._timeout, max_retries=0)
            return self._client

    # ------------------------------------------------------------------ cache

    def _cache_path(self, payload: dict[str, Any]) -> Path:
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        digest = hashlib.sha256(blob).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    # ------------------------------------------------------------------- call

    def json(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str = "response",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Call the model and return parsed JSON conforming to ``schema``."""
        schema = _strip_unsupported(schema)
        key = {
            "backend": self.backend,
            "model": self.model,
            "messages": messages,
            "schema": schema,
            "effort": self.effort if self._supports_effort else None,
        }
        cache_path = self._cache_path(key)
        if use_cache and cache_path.exists():
            self.usage.hit()
            return json.loads(cache_path.read_text())["response"]

        raw = self._call_with_retry(messages, schema, schema_name)
        parsed = json.loads(raw)
        if use_cache:
            cache_path.write_text(json.dumps({"model": self.model, "response": parsed}))
        return parsed

    def _call_with_retry(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], schema_name: str
    ) -> str:
        last: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.monotonic()
            try:
                if self.backend == "anthropic":
                    text = self._call_anthropic(messages, schema)
                else:
                    text = self._call_gateway(messages, schema, schema_name)
                return text
            except Exception as exc:  # noqa: BLE001 - re-raised below if fatal
                if not self._is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                last = exc
            finally:
                _ = time.monotonic() - start
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        raise RuntimeError(f"model call failed after {self.max_retries} attempts") from last

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            # Connection/timeout errors carry no status; those are worth a retry.
            return "connection" in type(exc).__name__.lower() or "timeout" in type(
                exc
            ).__name__.lower()
        # 402 (budget) and other 4xx are permanent; 429 and 5xx are not.
        return status == 429 or status >= 500

    # --------------------------------------------------------------- backends

    def _call_anthropic(self, messages: list[dict[str, Any]], schema: dict[str, Any]) -> str:
        system, turns = _split_system(messages)
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": schema}
        }
        # Not every model exposes the effort dial (Haiku, for one). Include it
        # only until the API tells us otherwise, then remember and stop trying.
        if self.effort and self._supports_effort:
            output_config["effort"] = self.effort

        # The system prompt is byte-identical on every routing call and is by
        # far the largest part of the request (2,863 tokens vs ~830 for the
        # per-message dossier). Caching it turns ~78% of the input into
        # cache reads at a tenth of the price. It cannot affect the decision:
        # the bytes the model sees are unchanged.
        system_blocks = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system
            else system
        )

        start = time.monotonic()
        try:
            response = self._backend_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=[_to_anthropic(m) for m in turns],
                output_config=output_config,
            )
        except Exception as exc:  # noqa: BLE001 - narrowed by the message check
            # Key off what *this* call sent, not the shared flag: with several
            # worker threads in flight, another thread may already have cleared
            # it, and testing the flag here would re-raise instead of retrying.
            if "effort" not in output_config or "effort" not in str(exc):
                raise
            self._supports_effort = False
            output_config.pop("effort", None)
            response = self._backend_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                messages=[_to_anthropic(m) for m in turns],
                output_config=output_config,
            )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model declined the request: {response.stop_details}")
        usage = response.usage
        self.usage.record(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            time.monotonic() - start,
            getattr(usage, "cache_creation_input_tokens", 0) or 0,
            getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        raise RuntimeError(f"no text block in response (stop_reason={response.stop_reason})")

    def _call_gateway(
        self, messages: list[dict[str, Any]], schema: dict[str, Any], schema_name: str
    ) -> str:
        start = time.monotonic()
        response = self._backend_client().chat.completions.create(
            model=self.model,
            messages=[_to_openai(m) for m in messages],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            },
        )
        usage = getattr(response, "usage", None)
        self.usage.record(
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
            time.monotonic() - start,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("empty completion")
        return content


# ------------------------------------------------------------------ shaping

def _split_system(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            content = message["content"]
            system_parts.append(
                content if isinstance(content, str) else _flatten_text(content)
            )
        else:
            turns.append(message)
    return "\n\n".join(system_parts), turns


def _flatten_text(content: list[dict[str, Any]]) -> str:
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def _to_anthropic(message: dict[str, Any]) -> dict[str, Any]:
    """Convert our neutral content blocks into Anthropic's shape."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}

    blocks: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            blocks.append({"type": "text", "text": block["text"]})
        elif kind == "image":
            blocks.append(block)
        elif kind == "image_url":
            # data:<mime>;base64,<payload>
            url = block["image_url"]["url"]
            header, _, payload = url.partition(",")
            media_type = header.split(":", 1)[1].split(";", 1)[0]
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": payload},
                }
            )
        # Anything else (e.g. inline audio) is unsupported here and dropped;
        # audio is resolved to text before it reaches the routing call.
    return {"role": message["role"], "content": blocks}


def _to_openai(message: dict[str, Any]) -> dict[str, Any]:
    content = message["content"]
    if isinstance(content, str):
        return message

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "image":
            source = block["source"]
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{source['media_type']};base64,{source['data']}"
                    },
                }
            )
        else:
            blocks.append(block)
    return {"role": message["role"], "content": blocks}
