"""MVT 2.3 / 3.4 — 上下文压缩策略

CompressionStrategy (ABC): 压缩策略抽象。
TruncateStrategy: 滑动窗口截断（Phase 2）。
SummarizeStrategy: LLM 摘要压缩（Phase 3 MVT 3.4）。
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

from src.agent.config import Config
from src.agent.models import Message
from src.agent.tokens import count_tokens


class CompressionStrategy(ABC):
    """压缩策略抽象"""

    @abstractmethod
    async def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        config: Config,
    ) -> list[Message]:
        """压缩消息列表

        Args:
            messages: 原始消息列表
            max_tokens: 最大 token 数
            config: 配置

        Returns:
            压缩后的消息列表（不修改原列表）
        """
        ...


class TruncateStrategy(CompressionStrategy):
    """滑动窗口截断：保留 system_prompt + 最近 N 条消息，丢弃中间最旧的消息"""

    async def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        config: Config,
    ) -> list[Message]:
        threshold = int(max_tokens * config.compression_threshold)
        keep_recent = config.compression_keep_recent

        if not messages:
            return []

        # 深拷贝，避免修改原列表
        msgs = copy.deepcopy(messages)

        # 分离 system 消息和非 system 消息
        system_msgs = [m for m in msgs if m.role == "system"]
        other_msgs = [m for m in msgs if m.role != "system"]

        # 始终保留 system prompt
        result: list[Message] = list(system_msgs)

        # 如果非 system 消息为空，直接返回
        if not other_msgs:
            return result

        # 确保至少保留最近 N 条
        if len(other_msgs) <= keep_recent:
            return system_msgs + other_msgs

        # 从最旧的非 system 消息开始移除
        model = config.llm_model
        # 保留尾部 keep_recent 条
        recent = other_msgs[-keep_recent:]
        rest = other_msgs[:-keep_recent]

        # 从 rest 的头部开始移除，直到 token 数低于阈值
        current_tokens = count_tokens(system_msgs + rest + recent, model)

        while rest and current_tokens > threshold:
            rest.pop(0)  # 移除最旧的
            current_tokens = count_tokens(system_msgs + rest + recent, model)

        return system_msgs + rest + recent


class SummarizeStrategy(CompressionStrategy):
    """LLM 摘要压缩（Phase 3 MVT 3.4）

    调用 LLM 对早期消息做摘要，保留为一条 system 消息。
    """

    def __init__(self, llm_client) -> None:
        from src.agent.llm.base import LLMClient
        self._llm: LLMClient = llm_client

    async def compress(
        self,
        messages: list[Message],
        max_tokens: int,
        config: Config,
    ) -> list[Message]:
        threshold = int(max_tokens * config.compression_threshold)
        keep_recent = config.compression_keep_recent

        if not messages:
            return []

        msgs = copy.deepcopy(messages)

        system_msgs = [m for m in msgs if m.role == "system"]
        other_msgs = [m for m in msgs if m.role != "system"]

        if not other_msgs:
            return list(system_msgs)

        if len(other_msgs) <= keep_recent:
            return system_msgs + other_msgs

        # 需要压缩的消息
        to_summarize = other_msgs[:-keep_recent]
        recent = other_msgs[-keep_recent:]

        # 构建摘要请求
        conversation_text = "\n".join(
            f"[{m.role}]: {m.content}" for m in to_summarize
        )

        summary_prompt = (
            "Please summarize the following conversation history into a concise summary, "
            "preserving key facts, decisions, and context. Write the summary in English.\n\n"
            f"Conversation:\n{conversation_text}\n\nSummary:"
        )

        summary_messages = [
            Message(role="system", content="You are a conversation summarizer."),
            Message(role="user", content=summary_prompt),
        ]

        try:
            response = await self._llm.chat(summary_messages)
            summary_text = response.content or "[Summary unavailable]"
        except Exception:
            summary_text = "[Summary failed: conversation too long to summarize]"

        # 构造摘要消息
        summary_msg = Message(
            role="system",
            content=f"[Conversation History Summary]\n{summary_text}",
        )

        # 返回：[system_prompt] + [摘要] + [最近 N 条]
        return system_msgs + [summary_msg] + recent
