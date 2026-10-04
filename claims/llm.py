"""Thin Claude client used when ANTHROPIC_API_KEY (or another SDK credential) is present.

Every project runs end-to-end without it: the offline backends are deterministic so
experiments are reproducible. Set USE_CLAUDE=1 to route model calls through Claude.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

DEFAULT_MODEL = "claude-opus-5-5"
SMALL_MODEL = "claude-haiku-4-5"


def claude_enabled() -> bool:
    if os.environ.get("USE_CLAUDE") != "1":
        return False
    has_creds = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")) or \
        os.path.isdir(os.path.expanduser("~/.config/anthropic"))
    if not has_creds:
        raise SystemExit("USE_CLAUDE=1 but no Anthropic credentials were found. Set ANTHROPIC_API_KEY "
                         "(or run `ant auth login`), or unset USE_CLAUDE to use the offline backend.")
    return True


_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic()
    return _client


def call_json(
    system: str,
    content: List[Dict[str, Any]],
    schema: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    effort: str = "low",
    max_tokens: int = 16000,
) -> Dict[str, Any]:
    """One structured-output call. Returns parsed JSON plus usage metadata under '_usage'.

    Raises RuntimeError on refusal / truncation so callers can route to a human.
    """
    import anthropic

    client = _get_client()
    kwargs: Dict[str, Any] = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": content}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
    )
    try:
        if model == DEFAULT_MODEL:
            # Server-side refusal fallback: the API re-runs a declined request on a
            # fallback model chosen by refusal category.
            resp = client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
                **kwargs,
            )
        else:
            kwargs["output_config"] = {"format": kwargs["output_config"]["format"]}
            resp = client.messages.create(**kwargs)
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"rate_limited: {e}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"api_error {e.status_code}: {e}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"connection_error: {e}") from e

    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={resp.stop_reason}")
    text = next(b.text for b in resp.content if b.type == "text")
    out = json.loads(text)
    out["_usage"] = {
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "model": model,
    }
    return out


def pdf_or_image_block(path: str) -> Optional[Dict[str, Any]]:
    """Build a multimodal content block for a scanned claim (PNG/JPEG/PDF)."""
    import base64

    ext = os.path.splitext(path)[1].lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".pdf": "application/pdf"}.get(ext)
    if not media:
        return None
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode()
    kind = "document" if media == "application/pdf" else "image"
    return {"type": kind, "source": {"type": "base64", "media_type": media, "data": data}}
