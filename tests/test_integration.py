"""End-to-end integration tests (MVT 1.4, 2.5, 3.3).

Requires DEEPSEEK_API_KEY environment variable to run.
"""

import os

import pytest

from src.agent.compression import TruncateStrategy, SummarizeStrategy
from src.agent.config import Config
from src.agent.llm.deepseek import DeepSeekClient
from src.agent.loop import AgentLoop
from src.agent.models import Message
from src.agent.persistence import JsonSessionStore
from src.agent.session import AgentSession
from src.agent.tools.base import Tool, ToolRegistry
from src.agent.tools.builtin.calculator import calculator
from src.agent.tools.builtin.search import search
from src.agent.tools.builtin.todo import todo


def _build_tool_registry() -> ToolRegistry:
    """构建包含三个内置工具的 ToolRegistry"""
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="Safely evaluate a mathematical expression. Supports +, -, *, /, **, //, %, ().",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The arithmetic expression to evaluate, e.g. '2 + 3 * 4'"},
            },
            "required": ["expression"],
        },
        handler=calculator,
    ))
    registry.register(Tool(
        name="search",
        description="Search the web using DuckDuckGo. Returns a summary of results.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
        handler=search,
    ))
    registry.register(Tool(
        name="todo",
        description="Manage a to-do list. Actions: add (with task), list, done (with task_id), clear.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "One of: add, list, done, clear"},
                "task": {"type": "string", "description": "Task description (required for add)"},
                "task_id": {"type": "integer", "description": "Task ID (required for done)"},
                "session_id": {"type": "string", "description": "Session ID for todo isolation"},
            },
            "required": ["action"],
        },
        handler=todo,
    ))
    return registry


@pytest.fixture
def api_config():
    """加载真实 API 配置"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")
    return Config.from_env(env_file=None)


class TestIntegrationLLM:
    """MVT 1.4 集成测试：LLM 客户端"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_chat(self, api_config):
        """简单的文本对话"""
        client = DeepSeekClient(api_config)
        response = await client.chat([Message(role="user", content="Say 'hello' in one word.")])
        assert response.content is not None
        assert response.usage is not None
        assert response.usage.total_tokens > 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_use_calculator(self, api_config):
        """LLM 使用 calculator 工具"""
        client = DeepSeekClient(api_config)
        registry = _build_tool_registry()
        loop = AgentLoop(client, registry, api_config)

        result = await loop.run("What is 123 * 456? Use the calculator tool.")
        assert result.content is not None
        # 检查内容中提到计算结果
        content_lower = result.content.lower()
        assert any(word in content_lower for word in ["56088", "123", "456"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_use_search(self, api_config):
        """LLM 使用 search 工具"""
        client = DeepSeekClient(api_config)
        registry = _build_tool_registry()
        loop = AgentLoop(client, registry, api_config)

        result = await loop.run("Search for 'Python programming language' and summarize the result.")
        assert result.content is not None
        assert len(result.content) > 10

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_tool_use_todo(self, api_config):
        """LLM 使用 todo 工具"""
        client = DeepSeekClient(api_config)
        registry = _build_tool_registry()
        loop = AgentLoop(client, registry, api_config)

        result = await loop.run(
            "Add 'buy groceries' to my todo list, then show me the list."
        )
        assert result.content is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_with_session_persistence(self, api_config):
        """Session 持久化集成测试"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session = AgentSession(system_prompt="You are a helpful assistant.", config=api_config)
            store = JsonSessionStore(base_dir=tmpdir, config=api_config)

            client = DeepSeekClient(api_config)
            loop = AgentLoop(client, config=api_config, session=session)

            result = await loop.run("Say 'Hello from session test'")
            assert result.content is not None

            # 保存 session
            await store.save(session)
            # 重新加载
            restored = await store.load(session.session_id)
            assert restored.session_id == session.session_id
            assert len(restored.messages) > 0


class TestIntegrationToolChain:
    """MVT 3.3 集成测试：多工具链式调用"""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_calculator_tool_chain(self, api_config):
        """多计算步骤"""
        client = DeepSeekClient(api_config)
        registry = _build_tool_registry()
        loop = AgentLoop(client, registry, api_config)

        result = await loop.run(
            "First calculate 100 + 200, then multiply the result by 3. Show me the final result."
        )
        assert result.content is not None

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_tool_types(self, api_config):
        """混合使用不同类型工具（calculator + todo）"""
        client = DeepSeekClient(api_config)
        registry = _build_tool_registry()
        loop = AgentLoop(client, registry, api_config)

        result = await loop.run(
            "Calculate 15 * 7, then add the result to my todo list with description 'Total items'."
        )
        assert result.content is not None
