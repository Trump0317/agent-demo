"""Tests for Phase 4 — Agent 统一入口."""
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.agent import (
    Agent,
    ALL_BUILTIN_TOOLS,
    BUILTIN_CALCULATOR,
    BUILTIN_SEARCH,
    BUILTIN_TODO,
    builtin_tools,
    create_agent,
)
from src.agent.config import Config
from src.agent.models import LLMResponse, Message, Usage
from src.agent.persistence import JsonSessionStore
from src.agent.tools.base import Tool, ToolRegistry


@pytest.fixture
def config():
    return Config(llm_api_key="sk-test", max_iterations=2)


@pytest.fixture
def mock_llm_response():
    return LLMResponse(
        content="Mock response",
        finish_reason="stop",
        model="mock",
        usage=Usage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
    )


class TestBuiltinTools:
    """内置工具注册测试"""

    def test_builtin_tools_registry(self):
        reg = builtin_tools()
        assert isinstance(reg, ToolRegistry)
        schemas = reg.list_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "calculator" in names
        assert "search" in names
        assert "todo" in names

    def test_builtin_tools_constants(self):
        assert len(ALL_BUILTIN_TOOLS) == 3
        assert BUILTIN_CALCULATOR.name == "calculator"
        assert BUILTIN_SEARCH.name == "search"
        assert BUILTIN_TODO.name == "todo"


class TestAgentCreation:
    """Agent 创建测试"""

    def test_default_agent(self, config):
        agent = Agent(config=config)
        assert agent.tools == "builtin"
        assert agent.trace_enabled is True
        assert agent.system_prompt  # 默认不为空

    def test_agent_custom_system_prompt(self, config):
        agent = Agent(config=config, system_prompt="Custom prompt")
        assert agent.system_prompt == "Custom prompt"

    def test_agent_no_tools(self, config):
        agent = Agent(config=config, tools=None)
        assert agent.tools is None

    def test_agent_custom_tools(self, config):
        custom = [BUILTIN_CALCULATOR]
        agent = Agent(config=config, tools=custom)
        assert agent.tools == custom

    def test_agent_with_compression(self, config):
        agent = Agent(config=config, compression="truncate")
        assert agent.compression == "truncate"

    def test_agent_with_session_id(self, config):
        agent = Agent(config=config, session_id="my-session")
        assert agent.session_id == "my-session"


class TestAgentRun:
    """Agent.run() 测试"""

    @pytest.mark.asyncio
    async def test_run_simple_task(self, config, mock_llm_response):
        """Agent.run() 正确初始化并调用 Loop"""
        agent = Agent(config=config, tools=None, trace_enabled=False)

        # Mock DeepSeekClient.chat
        with patch("src.agent.agent.DeepSeekClient") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat = AsyncMock(return_value=mock_llm_response)

            result = await agent.run("Hello")

        assert result.content == "Mock response"
        assert agent._initialized
        assert len(agent.messages) > 0

    @pytest.mark.asyncio
    async def test_run_initializes_once(self, config, mock_llm_response):
        """Agent 延迟初始化仅执行一次"""
        agent = Agent(config=config, tools=None, trace_enabled=False)

        with patch("src.agent.agent.DeepSeekClient") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat = AsyncMock(return_value=mock_llm_response)

            await agent.run("First")
            await agent.run("Second")

        # DeepSeekClient 仅创建一次
        assert mock_cls.call_count == 1

    @pytest.mark.asyncio
    async def test_run_with_builtin_tools(self, config, mock_llm_response):
        """Agent 使用 builtin 工具集"""
        agent = Agent(config=config, trace_enabled=False)

        with patch("src.agent.agent.DeepSeekClient") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat = AsyncMock(return_value=mock_llm_response)

            result = await agent.run("Calculate 1+1")

        assert result.content == "Mock response"
        # 确认工具 schema 被传入
        call_args = mock_client.chat.call_args
        tools_arg = call_args[1].get("tools")
        assert tools_arg is not None
        tool_names = [t["function"]["name"] for t in tools_arg]
        assert "calculator" in tool_names

    @pytest.mark.asyncio
    async def test_run_persists_session(self, config, mock_llm_response):
        """Agent 在 run 后自动持久化 session"""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonSessionStore(base_dir=tmpdir, config=config)
            agent = Agent(
                config=config,
                tools=None,
                trace_enabled=False,
                session_store=store,
            )

            with patch("src.agent.agent.DeepSeekClient") as mock_cls:
                mock_client = mock_cls.return_value
                mock_client.chat = AsyncMock(return_value=mock_llm_response)

                await agent.run("Task")

            # 验证 session 文件已创建
            session_files = await store.list_sessions()
            assert len(session_files) == 1


class TestAgentProperties:
    """Agent 属性测试"""

    def test_messages_before_init(self, config):
        agent = Agent(config=config)
        assert agent.messages == []
        assert agent.token_count == 0

    @pytest.mark.asyncio
    async def test_reset(self, config, mock_llm_response):
        agent = Agent(config=config, tools=None, trace_enabled=False)

        with patch("src.agent.agent.DeepSeekClient") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat = AsyncMock(return_value=mock_llm_response)

            await agent.run("Task")
            msg_count_before = len(agent.messages)
            assert msg_count_before > 0

            await agent.reset()
            assert len(agent.messages) < msg_count_before

    @pytest.mark.asyncio
    async def test_add_tool(self, config, mock_llm_response):
        """动态注册新工具——run 时包含新注册的工具"""
        agent = Agent(config=config, tools=[BUILTIN_CALCULATOR], trace_enabled=False)

        with patch("src.agent.agent.DeepSeekClient") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.chat = AsyncMock(return_value=mock_llm_response)
            await agent.add_tool(BUILTIN_SEARCH)
            # add_tool 触发 _init 创建了 loop，但未调用 chat
            # 再执行 run 才会实际调用 chat
            await agent.run("Search something")

        call_args = mock_client.chat.call_args
        tools_arg = call_args[1].get("tools")
        tool_names = [t["function"]["name"] for t in tools_arg]
        assert "calculator" in tool_names
        assert "search" in tool_names


class TestCreateAgent:
    """create_agent 工厂函数测试"""

    def test_create_agent_defaults(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        agent = create_agent()
        assert isinstance(agent, Agent)
        assert agent.tools == "builtin"

    def test_create_agent_with_options(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        agent = create_agent(
            system_prompt="Custom",
            tools=None,
            compression="truncate",
            trace_enabled=False,
        )
        assert agent.system_prompt == "Custom"
        assert agent.tools is None
        assert agent.compression == "truncate"
        assert agent.trace_enabled is False

    def test_create_agent_with_persist(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        agent = create_agent(persist=True)
        assert agent.session_store is not None
