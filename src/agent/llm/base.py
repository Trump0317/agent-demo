"""MVT 1.4 — Agent Executor 抽象接口

LLMClient (ABC): LLM 调用抽象。
ToolExecutor (ABC): 工具执行抽象（Phase 1 用 Stub，Phase 3 由 ToolRegistry 实现）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.agent.models import LLMResponse, Message, ToolCall, ToolResult


class LLMClient(ABC):
    """LLM 调用抽象接口"""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """发送消息到 LLM，返回统一响应

        Args:
            messages: 对话历史
            tools: OpenAI function calling 格式的工具列表，None 表示无工具

        Returns:
            LLMResponse: 统一 LLM 响应
        """
        ...


class ToolExecutor(ABC):
    """工具执行抽象

    Phase 1 用 Stub 占位，Phase 3 由 ToolRegistry 实现。
    """

    @abstractmethod
    async def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """执行工具调用，返回 ToolResult 列表"""
        ...

    @abstractmethod
    def list_schemas(self) -> list[dict]:
        """列出所有工具的 OpenAI Schema，供 LLM 调用时传入"""
        ...
