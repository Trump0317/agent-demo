"""Tests for MVT 1.5 / 2.5 — Agent Loop."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.compression import TruncateStrategy
from src.agent.config import Config
from src.agent.exceptions import LoopError
from src.agent.llm.base import LLMClient
from src.agent.loop import AgentLoop
from src.agent.models import (
    LLMResponse,
    Message,
    ToolCall,
    ToolCallFunction,
    ToolResult,
    Usage,
)
from src.agent.session import AgentSession


class MockLLMClient(LLMClient):
    """可编程 Mock LLM 客户端"""

    def __init__(self, responses: list[LLMResponse] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.call_args: list[tuple] = []

    async def chat(self, messages, tools=None):
        self.call_args.append((messages, tools))
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        # 默认返回 stop
        self.call_count += 1
        return LLMResponse(content="Done", finish_reason="stop", model="mock")


class MockToolExecutor:
    """Mock 工具执行器"""

    def __init__(self, results=None):
        self.results = results or []
        self.execute_calls: list[list[ToolCall]] = []

    async def execute(self, tool_calls):
        self.execute_calls.append(tool_calls)
        if self.results:
            return self.results
        return [
            ToolResult(
                tool_call_id=tc.id,
                tool_name=tc.function.name,
                success=True,
                result=f"Result from {tc.function.name}",
            )
            for tc in tool_calls
        ]

    def list_schemas(self):
        return [{"type": "function", "function": {"name": "mock_tool", "description": "Mock"}}]


@pytest.fixture
def config():
    return Config(
        llm_api_key="sk-test",
        max_iterations=3,
    )


class TestAgentLoopDirectStop:
    """LLM 直接返回 stop 的场景"""

    @pytest.mark.asyncio
    async def test_simple_stop(self, config):
        """LLM 直接返回 stop：Loop 正常终止"""
        mock_llm = MockLLMClient([
            LLMResponse(content="Hello!", finish_reason="stop", model="mock")
        ])
        loop = AgentLoop(mock_llm, config=config)
        result = await loop.run("Hi")
        assert result.content == "Hello!"
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_with_system_prompt(self, config):
        """带 system prompt 的纯对话"""
        mock_llm = MockLLMClient([
            LLMResponse(content="I'm helpful.", finish_reason="stop", model="mock")
        ])
        loop = AgentLoop(mock_llm, config=config)
        result = await loop.run("Hi", system_prompt="You are a helpful assistant.")
        assert result.content == "I'm helpful."
        # 确保 system prompt 被传递
        msgs = mock_llm.call_args[0][0]
        assert msgs[0].role == "system"
        assert msgs[0].content == "You are a helpful assistant."


class TestAgentLoopWithTools:
    """LLM 返回 tool_calls 的场景"""

    @pytest.mark.asyncio
    async def test_single_tool_call(self, config):
        """一次 tool_call 后 stop"""
        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", function=ToolCallFunction(name="calc", arguments='{"expr":"1+1"}'))
                ],
                finish_reason="tool_calls",
                model="mock",
            ),
            LLMResponse(content="The result is 2", finish_reason="stop", model="mock"),
        ])
        mock_tools = MockToolExecutor()
        loop = AgentLoop(mock_llm, mock_tools, config=config)
        result = await loop.run("Calculate 1+1")

        assert result.content == "The result is 2"
        assert mock_llm.call_count == 2
        assert len(mock_tools.execute_calls) == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, config):
        """多次 tool_call 后 stop"""
        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", function=ToolCallFunction(name="search", arguments='{"q":"x"}'))
                ],
                finish_reason="tool_calls",
                model="mock",
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c2", function=ToolCallFunction(name="calc", arguments='{"expr":"2+2"}'))
                ],
                finish_reason="tool_calls",
                model="mock",
            ),
            LLMResponse(content="Final answer", finish_reason="stop", model="mock"),
        ])
        mock_tools = MockToolExecutor()
        loop = AgentLoop(mock_llm, mock_tools, config=config)
        result = await loop.run("Complex task")

        assert result.content == "Final answer"
        assert mock_llm.call_count == 3
        assert len(mock_tools.execute_calls) == 2


class TestAgentLoopMaxIterations:
    """超过 max_iterations 的场景"""

    @pytest.mark.asyncio
    async def test_exceeds_max_iterations(self, config):
        """超过最大迭代抛出 LoopError"""
        config.max_iterations = 2
        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id=f"c{i}", function=ToolCallFunction(name="loop", arguments="{}"))
                ],
                finish_reason="tool_calls",
                model="mock",
            )
            for i in range(5)
        ])
        loop = AgentLoop(mock_llm, MockToolExecutor(), config=config)
        with pytest.raises(LoopError, match="max iterations"):
            await loop.run("Infinite task")

    @pytest.mark.asyncio
    async def test_exactly_max_iterations_no_error(self, config):
        """在 max_iterations 次内 stop，不抛异常"""
        config.max_iterations = 2
        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", function=ToolCallFunction(name="search", arguments="{}"))
                ],
                finish_reason="tool_calls",
                model="mock",
            ),
            LLMResponse(content="Done", finish_reason="stop", model="mock"),
        ])
        loop = AgentLoop(mock_llm, MockToolExecutor(), config=config)
        result = await loop.run("Task")
        assert result.content == "Done"


class TestAgentLoopNoTools:
    """无工具时的纯对话场景"""

    @pytest.mark.asyncio
    async def test_no_tools_provided(self, config):
        """不传 tool_executor 时正常对话"""
        mock_llm = MockLLMClient([
            LLMResponse(content="I have no tools.", finish_reason="stop", model="mock")
        ])
        loop = AgentLoop(mock_llm, config=config)
        result = await loop.run("Can you calculate?")
        assert result.content == "I have no tools."

    @pytest.mark.asyncio
    async def test_tool_call_without_tools(self, config):
        """无工具时 LLM 返回 tool_calls（stub 处理）"""
        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", function=ToolCallFunction(name="calc", arguments="{}"))
                ],
                finish_reason="tool_calls",
                model="mock",
            ),
            LLMResponse(content="I can't, but here's what I think...", finish_reason="stop", model="mock"),
        ])
        loop = AgentLoop(mock_llm, config=config)
        result = await loop.run("Calculate 1+1")
        # stub 返回错误，LLM 收到错误后给了回复
        assert result.content is not None


# ============================================================
# Phase 2 Integration Tests (MVT 2.5)
# ============================================================


class TestAgentLoopWithSession:
    """AgentLoop 与 AgentSession 集成测试"""

    @pytest.mark.asyncio
    async def test_uses_session_messages(self, config):
        """使用已有 session 时可恢复对话历史"""
        session = AgentSession(system_prompt="You are helpful.", config=config)
        await session.add_user_task("Previous question")
        await session.add_message(Message(role="assistant", content="Previous answer"))

        mock_llm = MockLLMClient([
            LLMResponse(content="I remember our conversation.", finish_reason="stop", model="mock")
        ])
        loop = AgentLoop(mock_llm, config=config, session=session)
        result = await loop.run("New question")

        assert result.content == "I remember our conversation."
        # session 应包含至少 4 条消息：system + user + assistant + user(new) + assistant(new)
        msgs = session.messages
        assert len(msgs) >= 5

    @pytest.mark.asyncio
    async def test_session_accumulates_messages(self, config):
        """多轮对话中 session 正确累积消息"""
        session = AgentSession(config=config)

        mock_llm = MockLLMClient([
            LLMResponse(content="First response", finish_reason="stop", model="mock")
        ])
        loop = AgentLoop(mock_llm, config=config, session=session)
        await loop.run("Question 1")

        count1 = len(session.messages)

        mock_llm2 = MockLLMClient([
            LLMResponse(content="Second response", finish_reason="stop", model="mock")
        ])
        loop2 = AgentLoop(mock_llm2, config=config, session=session)
        await loop2.run("Question 2")

        assert len(session.messages) > count1

    @pytest.mark.asyncio
    async def test_compression_integration(self, config):
        """压缩集成：token 超阈时触发压缩"""
        config.compression_threshold = 0.01  # 极低阈值，确保触发
        config.compression_keep_recent = 1

        session = AgentSession(system_prompt="S", config=config)
        # 先添加一些消息增加 token 数
        for i in range(10):
            await session.add_message(Message(role="user", content=f"Message {i} " + "x" * 100))
            await session.add_message(Message(role="assistant", content=f"Reply {i} " + "y" * 100))

        mock_llm = MockLLMClient([
            LLMResponse(content="Compressed response", finish_reason="stop", model="mock",
                        usage=Usage(prompt_tokens=100, completion_tokens=5, total_tokens=105)),
        ])

        compression = TruncateStrategy()
        loop = AgentLoop(mock_llm, config=config, session=session, compression_strategy=compression)
        result = await loop.run("Final question")

        assert result.content == "Compressed response"
        # 压缩后消息应减少（尽管又加了新的 user+assistant）
        # 至少验证循环完成了


class TestAgentLoopWithSessionAndTools:
    """Session + Tool 组合集成测试"""

    @pytest.mark.asyncio
    async def test_session_with_tools(self, config):
        """Session 模式下 LLM 调用工具"""
        session = AgentSession(config=config)

        mock_llm = MockLLMClient([
            LLMResponse(
                tool_calls=[
                    ToolCall(id="c1", function=ToolCallFunction(name="mock_tool", arguments="{}"))
                ],
                finish_reason="tool_calls",
                model="mock",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            ),
            LLMResponse(content="Tool result processed", finish_reason="stop", model="mock",
                        usage=Usage(prompt_tokens=20, completion_tokens=5, total_tokens=25)),
        ])
        mock_tools = MockToolExecutor()
        loop = AgentLoop(mock_llm, mock_tools, config=config, session=session)
        result = await loop.run("Do something")

        assert result.content == "Tool result processed"
        # session 应包含工具消息
        msgs = session.messages
        tool_msgs = [m for m in msgs if m.role == "tool"]
        assert len(tool_msgs) >= 1
