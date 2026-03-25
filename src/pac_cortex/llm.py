"""Thin OpenAI wrapper for SGR-style structured completions."""

import logging
import re
import time
from typing import Any, TypeVar

import openai
from pydantic import BaseModel

from pac_cortex.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)
_MAX_RETRIES = 6


def _retry_delay_from_error(exc: openai.RateLimitError) -> float:
    """Extract suggested retry delay from Gemini/OpenAI rate-limit response."""
    m = _RETRY_DELAY_RE.search(str(exc))
    return float(m.group(1)) if m else 60.0


class LLMClient:
    def __init__(self) -> None:
        # max_retries=0: we do our own backoff so the SDK doesn't burn retries instantly
        self._client = openai.OpenAI(
            api_key=settings.llm_api_key or None,
            base_url=settings.openai_base_url or None,
            max_retries=0,
        )
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    def parse_step(
        self,
        messages: list[dict[str, Any]],
        response_format: type[T],
        max_completion_tokens: int = 16384,
    ) -> T:
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self._client.beta.chat.completions.parse(
                    model=settings.llm_model,
                    messages=messages,  # type: ignore[arg-type]
                    response_format=response_format,
                    max_completion_tokens=max_completion_tokens,
                )
                usage = resp.usage
                if usage:
                    self.total_prompt_tokens += usage.prompt_tokens
                    self.total_completion_tokens += usage.completion_tokens
                parsed = resp.choices[0].message.parsed
                if parsed is None:
                    raise RuntimeError("LLM returned no parsed response")
                return parsed
            except openai.RateLimitError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = _retry_delay_from_error(exc)
                logger.warning("Rate limit — waiting %.0fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
            except openai.APIError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                delay = 2.0 ** attempt
                logger.warning("API error %s — retrying in %.0fs", exc, delay)
                time.sleep(delay)

        raise RuntimeError("parse_step exhausted retries")  # unreachable
