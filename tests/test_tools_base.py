"""Tests for MVT 3.1 — Tool base (Tool and ToolRegistry)."""
import asyncio

import pytest

from src.agent.exceptions import ToolNotFoundError, ToolExecutionError
from src.agent.llm.base import ToolExecutor
from src.agent.models import ToolCall, ToolCallFunction
from src.agent.tools.base import Tool, ToolRegistry


# ── Test helper functions ──

async def _echo(text: str) -> str:
    return f"Echo: {text}"


async def _add(a: float, b: float) -> str:
    return str(a + b)


async def _slow_op(delay: float = 5) -> str:
    await asyncio.sleep(delay)
    return "done"


async def _failing_op() -> str:
    raise ValueError("Intentional failure")


# ── Fixtures ──

@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(Tool(
        name="echo",
        description="Echo back the input",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "Text to echo"}},
            "required": ["text"],
        },
        handler=_echo,
    ))
    reg.register(Tool(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        handler=_add,
    ))
    return reg


class TestTool:
    """Tool 数据类测试"""

    def test_to_openai_schema(self):
        tool = Tool(
            name="test",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            handler=_echo,
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"
        assert schema["function"]["description"] == "A test tool"


class TestToolRegistry:
    """ToolRegistry 测试"""

    def test_register_and_get(self, registry):
        tool = registry.get("echo")
        assert tool.name == "echo"
        assert tool.description == "Echo back the input"

    def test_get_nonexistent_raises(self, registry):
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            registry.get("nonexistent")

    def test_list_schemas(self, registry):
        schemas = registry.list_schemas()
        assert len(schemas) == 2
        names = [s["function"]["name"] for s in schemas]
        assert "echo" in names
        assert "add" in names

    def test_is_tool_executor(self, registry):
        """ToolRegistry 继承 ToolExecutor"""
        assert isinstance(registry, ToolExecutor)

    @pytest.mark.asyncio
    async def test_execute_success(self, registry):
        """execute 正确调用工具并返回 ToolResult"""
        tc = ToolCall(id="c1", function=ToolCallFunction(name="echo", arguments='{"text": "hello"}'))
        results = await registry.execute([tc])
        assert len(results) == 1
        assert results[0].success
        assert results[0].result == "Echo: hello"
        assert results[0].tool_call_id == "c1"

    @pytest.mark.asyncio
    async def test_execute_with_numbers(self, registry):
        """数值参数工具"""
        tc = ToolCall(id="c2", function=ToolCallFunction(name="add", arguments='{"a": 3, "b": 4}'))
        results = await registry.execute([tc])
        assert results[0].success
        assert results[0].result == "7"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, registry):
        """调用不存在的工具"""
        tc = ToolCall(id="c3", function=ToolCallFunction(name="no_such_tool", arguments="{}"))
        results = await registry.execute([tc])
        assert not results[0].success
        assert "not found" in results[0].result.lower() or "Tool not found" in results[0].result

    @pytest.mark.asyncio
    async def test_execute_invalid_json(self, registry):
        """参数不是合法 JSON"""
        tc = ToolCall(id="c4", function=ToolCallFunction(name="echo", arguments="not json"))
        results = await registry.execute([tc])
        assert not results[0].success
        assert "json" in results[0].result.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_failure(self):
        """工具内部抛出异常"""
        reg = ToolRegistry()
        reg.register(Tool(
            name="fail",
            description="Always fails",
            parameters={"type": "object", "properties": {}},
            handler=_failing_op,
        ))
        tc = ToolCall(id="c5", function=ToolCallFunction(name="fail", arguments="{}"))
        results = await reg.execute([tc])
        assert not results[0].success

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self):
        """工具执行超时"""
        reg = ToolRegistry()
        reg.register(Tool(
            name="slow",
            description="Slow operation",
            parameters={
                "type": "object",
                "properties": {"delay": {"type": "number"}},
            },
            handler=_slow_op,
        ))
        # 设置工具超时为 0.1s，但 handler 会 sleep 5s
        tc = ToolCall(id="c6", function=ToolCallFunction(name="slow", arguments='{"delay": 5}'))
        tool = reg.get("slow")
        tool.timeout = 0.1  # 短超时
        results = await reg.execute([tc])
        assert not results[0].success
        assert "timed out" in results[0].result.lower()

    @pytest.mark.asyncio
    async def test_execute_batch(self, registry):
        """批量执行：部分成功部分失败"""
        tcs = [
            ToolCall(id="c1", function=ToolCallFunction(name="echo", arguments='{"text": "ok"}')),
            ToolCall(id="c2", function=ToolCallFunction(name="nonexistent", arguments="{}")),
            ToolCall(id="c3", function=ToolCallFunction(name="add", arguments='{"a": 1, "b": 2}')),
        ]
        results = await registry.execute(tcs)
        assert len(results) == 3
        assert results[0].success  # echo ok
        assert not results[1].success  # nonexistent
        assert results[2].success  # add ok
