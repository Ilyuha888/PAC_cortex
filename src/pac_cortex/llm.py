"""Thin wrapper around OpenAI SDK for chat completions with tool use."""

from typing import Any

import openai

from pac_cortex.config import settings


class LLMClient:
    def __init__(self) -> None:
        self._client = openai.AsyncOpenAI(api_key=settings.llm_api_key)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Send a chat completion request. Returns parsed response with tool_calls or content."""
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        last_error: Exception | None = None
        for _ in range(max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except openai.RateLimitError as e:
                last_error = e
                continue
            except openai.APIError as e:
                last_error = e
                continue
        else:
            msg = f"LLM request failed after {max_retries + 1} attempts"
            raise RuntimeError(msg) from last_error

        choice = response.choices[0]
        usage = response.usage
        if usage:
            self.total_prompt_tokens += usage.prompt_tokens
            self.total_completion_tokens += usage.completion_tokens

        if choice.message.tool_calls:
            return {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in choice.message.tool_calls
                ],
                "raw_message": choice.message,
            }

        return {
            "type": "text",
            "content": choice.message.content or "",
            "raw_message": choice.message,
        }
