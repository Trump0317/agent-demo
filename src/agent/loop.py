"""MVT 1.5 / 2.5 — Agent Loop

Agent 主循环：接收 task，调用 LLM，处理工具调用，迭代直到完成。
Phase 2 集成：AgentSession + CompressionStrategy。
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from src.agent.config import Config
from src.agent.exceptions import LoopError
from src.agent.llm.base import LLMClient, ToolExecutor
from src.agent.models import (
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    TracePhase,
)

if TYPE_CHECKING:
    from src.agent.compression import CompressionStrategy
    from src.agent.session import AgentSession
    from src.agent.trace import TraceCollector

logger = logging.getLogger(__name__)

# Phase 1 Stub ToolExecutor
class _StubToolExecutor(ToolExecutor):
    """Phase 1 占位：无工具的 ToolExecutor"""

    async def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        return [
            ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.function.name,
                success=False,
                result=f"Error: tool '{tc.function.name}' not available (stub)",
            )
            for tc in tool_calls
        ]

    def list_schemas(self) -> list[dict]:
        return []


class AgentLoop:
    """Agent 主循环

    编排 LLM 调用与工具执行，支持会话管理和上下文压缩（Phase 2+）。

    Phase 2 新增参数：
    - session: 可选 AgentSession，用于管理对话历史和 token 计数
    - compression_strategy: 可选上下文压缩策略
    """

    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor | None = None,
        config: Config | None = None,
        trace_collector: "TraceCollector | None" = None,
        session: "AgentSession | None" = None,
        compression_strategy: "CompressionStrategy | None" = None,
    ):
        self._llm = llm_client
        self._tools = tool_executor or _StubToolExecutor()
        self._config = config or Config()
        self._trace = trace_collector
        self._session = session
        self._compression = compression_strategy

    @property
    def tool_executor(self) -> ToolExecutor:
        return self._tools

    @property
    def config(self) -> Config:
        return self._config

    async def run(
        self,
        task: str,
        *,
        messages: list[Message] | None = None,
        system_prompt: str = "",
    ) -> LLMResponse:
        """执行一次 Agent Loop

        Args:
            task: 用户任务
            messages: 已有消息历史。若提供了 session，则优先使用 session 的消息历史
            system_prompt: 系统提示词（仅在无 session 时生效；有 session 时使用 session 的 system_prompt）

        Returns:
            LLMResponse: LLM 的最终响应

        Raises:
            LoopError: 超过最大迭代次数仍未完成
        """
        # 确定消息来源：session > messages 参数 > 新建
        use_session = self._session is not None

        if use_session:
            # Phase 2: 使用 AgentSession 管理对话历史
            await self._session.add_user_task(task)
            msgs: list[Message] = self._session.messages
        else:
            # Phase 1: 使用临时消息列表
            if messages is None:
                msgs = []
            else:
                msgs = list(messages)

            if system_prompt:
                if not msgs or msgs[0].role != "system":
                    msgs.insert(0, Message(role="system", content=system_prompt))

            # 追加 user 任务
            msgs.append(Message(role="user", content=task))

        tools_schemas = self._tools.list_schemas() or None

        for iteration in range(1, self._config.max_iterations + 1):
            # ── Phase 2: 压缩检查 ──
            if use_session and self._compression is not None:
                threshold_ratio = self._session.token_count / self._config.max_context_tokens
                if threshold_ratio > self._config.compression_threshold:
                    if self._trace:
                        self._trace.record(
                            TracePhase.COMPRESSION,
                            {
                                "strategy": type(self._compression).__name__,
                                "before_tokens": self._session.token_count,
                                "after_tokens": 0,  # 将在压缩后更新
                                "removed_count": 0,
                            },
                        )
                    msgs = await self._compression.compress(
                        msgs, self._config.max_context_tokens, self._config
                    )
                    self._session.replace_messages(msgs)
                    msgs = self._session.messages  # 使用更新后的消息

                    if self._trace:
                        self._trace.record(
                            TracePhase.COMPRESSION,
                            {
                                "strategy": type(self._compression).__name__,
                                "before_tokens": self._session.token_count,
                                "after_tokens": self._session.token_count,
                                "removed_count": 0,
                            },
                        )

            # Trace: loop iteration
            if self._trace:
                self._trace.record(
                    TracePhase.LOOP_ITERATION,
                    {
                        "iteration": iteration,
                        "max_iterations": self._config.max_iterations,
                        "total_tokens_so_far": (
                            self._session.token_count if use_session else len(str(msgs))
                        ),
                    },
                )

            # Trace: LLM request
            if self._trace:
                self._trace.record(
                    TracePhase.LLM_REQUEST,
                    {
                        "model": self._config.llm_model,
                        "message_count": len(msgs),
                        "tools": [t.get("function", {}).get("name", "") for t in (tools_schemas or [])],
                        "estimated_tokens": (
                            self._session.token_count if use_session else len(str(msgs))
                        ),
                    },
                )

            t_start = time.monotonic()
            response = await self._llm.chat(msgs, tools=tools_schemas)
            latency_ms = (time.monotonic() - t_start) * 1000

            # ── Phase 2: 用 LLM 返回的 usage 更新 session token 计数 ──
            if use_session and response.usage:
                self._session.set_token_count(response.usage.total_tokens)

            # Trace: LLM response
            if self._trace:
                content_snippet = None
                if response.content:
                    content_snippet = response.content[:self._config.trace_content_max_length]
                tc_names = None
                if response.tool_calls:
                    tc_names = [tc.function.name for tc in response.tool_calls]
                self._trace.record(
                    TracePhase.LLM_RESPONSE,
                    {
                        "content": content_snippet,
                        "tool_calls": tc_names,
                        "usage": response.usage.to_dict() if response.usage else None,
                        "finish_reason": response.finish_reason,
                        "latency_ms": round(latency_ms, 2),
                    },
                )

            # 判断终止
            if response.finish_reason == "stop":
                # 追加 assistant 消息到 session
                if use_session and response.content:
                    await self._session.add_message(
                        Message(role="assistant", content=response.content)
                    )
                return response

            if response.finish_reason == "tool_calls" and response.tool_calls:
                # 执行工具
                for tc in response.tool_calls:
                    if self._trace:
                        self._trace.record(
                            TracePhase.TOOL_CALL,
                            {"tool_name": tc.function.name, "arguments": tc.function.arguments},
                        )

                t_tool_start = time.monotonic()
                tool_results = await self._tools.execute(response.tool_calls)
                tool_latency_ms = (time.monotonic() - t_tool_start) * 1000

                # 追加 assistant 消息（含 tool_calls）
                assistant_msg = Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=response.tool_calls,
                )

                if use_session:
                    await self._session.add_message(assistant_msg)
                else:
                    msgs.append(assistant_msg)

                # 追加 tool 结果消息
                for tr in tool_results:
                    # Trace: tool result
                    if self._trace:
                        self._trace.record(
                            TracePhase.TOOL_RESULT,
                            {
                                "tool_name": tr.tool_name,
                                "success": tr.success,
                                "result": tr.result[:self._config.trace_content_max_length],
                                "latency_ms": tr.latency_ms or round(tool_latency_ms / len(tool_results), 2),
                            },
                        )
                    tr.latency_ms = tr.latency_ms or round(tool_latency_ms / len(tool_results), 2)
                    tool_msg = tr.to_message()
                    if use_session:
                        await self._session.add_message(tool_msg)
                    else:
                        msgs.append(tool_msg)

                # 刷新消息列表引用
                if use_session:
                    msgs = self._session.messages

                continue  # 下一轮迭代

            # 非预期的 finish_reason（例如 "length"），终止并返回
            return response

        # 超过最大迭代数
        raise LoopError(
            f"Agent loop exceeded max iterations ({self._config.max_iterations}). "
            f"Last finish_reason: {response.finish_reason or 'unknown'}."
        )
