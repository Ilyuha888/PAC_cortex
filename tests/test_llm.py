"""Tests for pac_cortex.llm — LLMClient structured completions."""

from unittest.mock import MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from pac_cortex.llm import LLMClient, _retry_delay_from_error


class _Step(BaseModel):
    value: str


class _FakeRateLimitError(Exception):
    pass


class _FakeAPIError(Exception):
    pass


def _make_resp(prompt_tokens: int = 10, completion_tokens: int = 5, parsed=None):
    resp = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.choices[0].message.parsed = parsed if parsed is not None else _Step(value="ok")
    return resp


@pytest.fixture
def llm_client():
    """LLMClient with openai.OpenAI and settings patched."""
    with (
        patch("pac_cortex.llm.openai.OpenAI") as mock_cls,
        patch("pac_cortex.llm.settings") as mock_settings,
    ):
        mock_settings.llm_api_key = "test-key"
        mock_settings.openai_base_url = ""
        mock_settings.llm_model = "test-model"
        mock_openai = MagicMock()
        mock_cls.return_value = mock_openai
        yield LLMClient(), mock_openai


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_parse_step_returns_parsed_object(llm_client) -> None:
    client, mock_openai = llm_client
    parsed = _Step(value="result")
    mock_openai.beta.chat.completions.parse.return_value = _make_resp(10, 5, parsed)

    result = client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert result is parsed


def test_parse_step_accumulates_tokens(llm_client) -> None:
    client, mock_openai = llm_client
    mock_openai.beta.chat.completions.parse.return_value = _make_resp(100, 50)

    client.parse_step([{"role": "user", "content": "first"}], _Step)
    client.parse_step([{"role": "user", "content": "second"}], _Step)

    assert client.total_prompt_tokens == 200
    assert client.total_completion_tokens == 100


def test_parse_step_no_usage_skips_token_accumulation(llm_client) -> None:
    client, mock_openai = llm_client
    resp = MagicMock()
    resp.usage = None
    resp.choices[0].message.parsed = _Step(value="ok")
    mock_openai.beta.chat.completions.parse.return_value = resp

    client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert client.total_prompt_tokens == 0
    assert client.total_completion_tokens == 0


def test_parse_step_none_parsed_raises_runtime_error(llm_client) -> None:
    client, mock_openai = llm_client
    resp = MagicMock()
    resp.usage = None
    resp.choices[0].message.parsed = None
    mock_openai.beta.chat.completions.parse.return_value = resp

    with pytest.raises(RuntimeError, match="no parsed response"):
        client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert mock_openai.beta.chat.completions.parse.call_count == 1


# ---------------------------------------------------------------------------
# Rate-limit retry
# ---------------------------------------------------------------------------


def test_parse_step_rate_limit_retries_then_succeeds(llm_client) -> None:
    client, mock_openai = llm_client
    parsed = _Step(value="ok")
    resp = _make_resp(parsed=parsed)

    with (
        patch.object(openai, "RateLimitError", _FakeRateLimitError),
        patch("pac_cortex.llm.time.sleep") as mock_sleep,
    ):
        mock_openai.beta.chat.completions.parse.side_effect = [
            _FakeRateLimitError("rate limit: retry in 2s"),
            resp,
        ]
        result = client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert result is parsed
    mock_sleep.assert_called_once_with(2.0)


def test_parse_step_rate_limit_exhausted_propagates(llm_client) -> None:
    client, mock_openai = llm_client

    with (
        patch.object(openai, "RateLimitError", _FakeRateLimitError),
        patch("pac_cortex.llm.time.sleep"),
    ):
        mock_openai.beta.chat.completions.parse.side_effect = _FakeRateLimitError("retry in 1s")

        with pytest.raises(_FakeRateLimitError):
            client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert mock_openai.beta.chat.completions.parse.call_count == 7  # _MAX_RETRIES + 1


# ---------------------------------------------------------------------------
# APIError retry
# ---------------------------------------------------------------------------


def test_parse_step_api_error_retries_with_backoff(llm_client) -> None:
    client, mock_openai = llm_client
    parsed = _Step(value="ok")
    resp = _make_resp(parsed=parsed)

    with (
        patch.object(openai, "RateLimitError", _FakeRateLimitError),
        patch.object(openai, "APIError", _FakeAPIError),
        patch("pac_cortex.llm.time.sleep") as mock_sleep,
    ):
        mock_openai.beta.chat.completions.parse.side_effect = [
            _FakeAPIError("server error"),
            resp,
        ]
        result = client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert result is parsed
    mock_sleep.assert_called_once_with(1.0)  # 2.0 ** 0


def test_parse_step_unknown_error_propagates_immediately(llm_client) -> None:
    client, mock_openai = llm_client
    mock_openai.beta.chat.completions.parse.side_effect = ValueError("unexpected")

    with pytest.raises(ValueError, match="unexpected"):
        client.parse_step([{"role": "user", "content": "go"}], _Step)

    assert mock_openai.beta.chat.completions.parse.call_count == 1


# ---------------------------------------------------------------------------
# _retry_delay_from_error
# ---------------------------------------------------------------------------


def test_retry_delay_extracts_float_seconds() -> None:
    exc = _FakeRateLimitError("too many requests: retry in 42.5s")
    assert _retry_delay_from_error(exc) == 42.5  # type: ignore[arg-type]


def test_retry_delay_extracts_integer_seconds() -> None:
    exc = _FakeRateLimitError("rate limit exceeded. Please retry in 30s.")
    assert _retry_delay_from_error(exc) == 30.0  # type: ignore[arg-type]


def test_retry_delay_fallback_when_no_match() -> None:
    exc = _FakeRateLimitError("rate limit exceeded, no ETA")
    assert _retry_delay_from_error(exc) == 60.0  # type: ignore[arg-type]
