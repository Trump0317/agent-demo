"""MVT 3.1 — Tool 基座

Tool: 工具定义（name, description, parameters, handler）。
ToolRegistry: 工具注册中心，实现 ToolExecutor 接口。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from src.agent.exceptions import ToolExecutionError, ToolNotFoundError
from src.agent.llm.base import ToolExecutor
from src.agent.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class Tool:
    """工具定义

    Attributes:
        name: 工具名称（唯一标识）
        description: 工具描述（给 LLM 看）
        parameters: JSON Schema 格式的参数定义
        handler: 异步处理函数，接收关键字参数，返回任意结果
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Awaitable[Any]]  # async function

    def to_openai_schema(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry(ToolExecutor):
    """工具注册中心

    继承 ToolExecutor（Phase 1 接口），可无缝替换 Stub。
    """

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, tool: Tool) -> None:
        """注册工具，同名覆盖"""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool:
        """获取工具

        Raises:
            ToolNotFoundError: 工具未注册
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return self._tools[name]

    def list_schemas(self) -> list[dict]:
        """列出所有工具的 OpenAI Schema"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    async def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """批量执行工具调用

        单个工具失败不影响其他工具，失败的 ToolResult.success=False。
        超时控制：使用 asyncio.wait_for，超时抛出 ToolExecutionError。

        Args:
            tool_calls: 待执行的工具调用列表

        Returns:
            ToolResult 列表，与输入顺序对应
        """
        import json
        import time

        results: list[ToolResult] = []

        for tc in tool_calls:
            t_start = time.monotonic()

            try:
                tool = self.get(tc.function.name)
            except ToolNotFoundError as e:
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    success=False,
                    result=f"Error: {e}",
                    latency_ms=(time.monotonic() - t_start) * 1000,
                ))
                continue

            try:
                # 解析参数
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    success=False,
                    result=f"Error: invalid JSON arguments — {e}",
                    latency_ms=(time.monotonic() - t_start) * 1000,
                ))
                continue

            try:
                # 超时执行
                timeout = getattr(tool, 'timeout', 30)
                result_value = await asyncio.wait_for(
                    tool.handler(**arguments),
                    timeout=timeout,
                )
                latency = (time.monotonic() - t_start) * 1000
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    success=True,
                    result=str(result_value),
                    latency_ms=round(latency, 2),
                ))
            except asyncio.TimeoutError:
                latency = (time.monotonic() - t_start) * 1000
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    success=False,
                    result=f"Error: tool '{tc.function.name}' timed out after {timeout}s",
                    latency_ms=round(latency, 2),
                ))
            except Exception as e:
                latency = (time.monotonic() - t_start) * 1000
                results.append(ToolResult(
                    tool_call_id=tc.id,
                    tool_name=tc.function.name,
                    success=False,
                    result=f"Error executing '{tc.function.name}': {e}",
                    latency_ms=round(latency, 2),
                ))

        return results
