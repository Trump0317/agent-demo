"""MVT 1.4 — DeepSeek LLM 客户端实现

基于 openai.AsyncOpenAI，兼容 OpenAI 接口。
"""

from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI, APIStatusError, APITimeoutError, AuthenticationError, RateLimitError

from src.agent.config import Config
from src.agent.exceptions import (
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.agent.llm.base import LLMClient
from src.agent.models import LLMResponse, Message, ToolCall, ToolCallFunction, Usage

logger = logging.getLogger(__name__)


class DeepSeekClient(LLMClient):
    """DeepSeek LLM 客户端（兼容 OpenAI 接口的其他模型也可使用）"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = AsyncOpenAI(
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            timeout=float(config.llm_timeout),
            max_retries=config.llm_max_retries,
        )

    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """发送消息到 DeepSeek/兼容 API，返回统一 LLMResponse"""
        openai_messages = [m.to_openai_dict() for m in messages]
        kwargs: dict = {
            "model": self.config.llm_model,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except AuthenticationError as e:
            raise LLMAuthError(f"LLM authentication failed: {e}") from e
        except RateLimitError as e:
            raise LLMRateLimitError(f"LLM rate limit exceeded: {e}") from e
        except APITimeoutError as e:
            raise LLMTimeoutError(f"LLM request timed out: {e}") from e
        except APIStatusError as e:
            raise LLMError(f"LLM API error ({e.status_code}): {e.message}") from e
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

        choice = response.choices[0]
        message = choice.message

        # 构建 LLMResponse
        content = message.content
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type=tc.type,
                    function=ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.finish_reason,
            model=response.model,
        )
