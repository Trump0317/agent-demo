"""Tests for MVT 2.2 — AgentSession."""
import pytest

from src.agent.config import Config
from src.agent.models import Message, ToolCall, ToolCallFunction
from src.agent.persistence import JsonSessionStore
from src.agent.session import AgentSession


@pytest.fixture
def config():
    return Config(llm_api_key="sk-test")


class TestAgentSessionCreation:
    """Session 创建测试"""

    def test_default_creation(self, config):
        session = AgentSession(config=config)
        assert session.session_id is not None
        assert len(session.messages) == 0
        assert session.token_count == 0

    def test_with_system_prompt(self, config):
        session = AgentSession(system_prompt="You are helpful.", config=config)
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"
        assert session.messages[0].content == "You are helpful."
        assert session.token_count > 0

    def test_custom_session_id(self, config):
        session = AgentSession(session_id="my-id", config=config)
        assert session.session_id == "my-id"


class TestAddMessage:
    """add_message 测试"""

    @pytest.mark.asyncio
    async def test_add_message_updates_token_count(self, config):
        session = AgentSession(config=config)
        initial_tokens = session.token_count
        await session.add_message(Message(role="user", content="Hello"))
        assert session.token_count > initial_tokens

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, config):
        session = AgentSession(config=config)
        await session.add_message(Message(role="user", content="First"))
        await session.add_message(Message(role="assistant", content="Second"))
        assert len(session.messages) == 2

    @pytest.mark.asyncio
    async def test_add_user_task(self, config):
        session = AgentSession(config=config)
        await session.add_user_task("Do something")
        assert len(session.messages) == 1
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Do something"


class TestReplaceMessages:
    """replace_messages 测试"""

    def test_replace_messages_token_recalc(self, config):
        session = AgentSession(system_prompt="You are helpful.", config=config)
        old_count = session.token_count

        new_msgs = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="A much longer message that should change token count significantly."),
        ]
        session.replace_messages(new_msgs)
        assert session.token_count != old_count

    def test_replace_messages_preserves_count(self, config):
        session = AgentSession(config=config)
        msgs = [
            Message(role="system", content="S"),
            Message(role="user", content="U"),
        ]
        session.replace_messages(msgs)
        count1 = session.token_count
        # 相同内容再次替换，计数相同
        session.replace_messages(msgs)
        assert session.token_count == count1


class TestReset:
    """reset 测试"""

    @pytest.mark.asyncio
    async def test_reset_keeps_system_prompt(self, config):
        session = AgentSession(system_prompt="Keep me", config=config)
        await session.add_user_task("Some task")
        assert len(session.messages) > 1

        await session.reset()
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"
        assert session.messages[0].content == "Keep me"

    @pytest.mark.asyncio
    async def test_reset_no_system_prompt(self, config):
        session = AgentSession(config=config)
        await session.add_user_task("Task")
        await session.reset()
        assert len(session.messages) == 0


class TestSessionSerialization:
    """序列化/反序列化测试"""

    def test_roundtrip(self, config):
        session = AgentSession(
            session_id="sid-1",
            system_prompt="You are helpful.",
            config=config,
        )
        # 手动添加（绕过 async）
        session._messages.append(Message(role="user", content="Hello"))
        session._recalc_tokens()

        d = session.to_dict()
        restored = AgentSession.from_dict(d, config=config)

        assert restored.session_id == "sid-1"
        assert restored.token_count == session.token_count
        assert len(restored.messages) == len(session.messages)
        assert restored.messages[0].role == "system"


class TestMultiSessionIsolation:
    """多窗口 Session 隔离测试——用户 A 开两个窗口互不影响"""

    @pytest.mark.asyncio
    async def test_two_sessions_independent(self, config):
        """两个 session 的历史完全独立"""
        session1 = AgentSession(session_id="window-1", config=config)
        session2 = AgentSession(session_id="window-2", config=config)

        # 窗口 1：查天气
        await session1.add_user_task("What's the weather in Beijing?")
        await session1.add_message(Message(role="assistant", content="Sunny, 25°C"))

        # 窗口 2：写周报
        await session2.add_user_task("Write weekly report summary")
        await session2.add_message(Message(role="assistant", content="Report drafted."))

        # 窗口 1 追问
        await session1.add_user_task("Should I bring an umbrella?")
        await session1.add_message(Message(role="assistant", content="No, it's sunny."))

        # 验证：窗口 1 有完整历史
        msgs1 = session1.messages
        assert len(msgs1) == 4  # user1 + asst1 + user2 + asst2 (no system prompt)
        assert "weather" in msgs1[0].content  # 第一次提问
        assert "umbrella" in msgs1[2].content  # 追问

        # 验证：窗口 2 看不到窗口 1 的内容
        msgs2 = session2.messages
        assert len(msgs2) == 2  # user1 + asst1
        assert "report" in msgs2[0].content
        assert "weather" not in str(msgs2)  # 不应出现窗口 1 的内容

    @pytest.mark.asyncio
    async def test_two_sessions_different_tools(self, config):
        """两个 session 使用不同工具集"""
        session1 = AgentSession(session_id="window-1", config=config)
        session2 = AgentSession(session_id="window-2", config=config)

        # 窗口 1：用 calculator
        await session1.add_user_task("Calculate 100 + 200")
        tc = ToolCall(id="c1", function=ToolCallFunction(name="calculator", arguments='{"expression":"100+200"}'))
        await session1.add_message(Message(role="assistant", content="", tool_calls=[tc]))
        await session1.add_message(Message(role="tool", content="300", tool_call_id="c1", name="calculator"))
        await session1.add_message(Message(role="assistant", content="The result is 300."))

        # 窗口 2：用 todo
        await session2.add_user_task("Add 'buy milk' to todo")
        tc2 = ToolCall(id="c2", function=ToolCallFunction(name="todo", arguments='{"action":"add","task":"buy milk"}'))
        await session2.add_message(Message(role="assistant", content="", tool_calls=[tc2]))
        await session2.add_message(Message(role="tool", content="Added todo #1", tool_call_id="c2", name="todo"))
        await session2.add_message(Message(role="assistant", content="Done!"))

        # 验证隔离
        msgs1 = session1.messages
        msgs2 = session2.messages
        assert "calculator" in str(msgs1)
        assert "todo" not in str(msgs1)
        assert "todo" in str(msgs2)
        assert "calculator" not in str(msgs2)

    @pytest.mark.asyncio
    async def test_session_persist_and_resume(self, config):
        """窗口 1 持久化后，新 Agent 恢复继续追问"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonSessionStore(base_dir=tmpdir, config=config)

            # 第一轮：窗口 1 查天气
            session = AgentSession(session_id="window-1", config=config)
            await session.add_user_task("Weather?")
            await session.add_message(Message(role="assistant", content="Sunny."))
            await store.save(session)

            # 第二轮：恢复窗口 1，追问
            restored = await store.load("window-1")
            assert restored.session_id == "window-1"
            msgs = restored.messages
            assert "Weather?" in msgs[0].content
            assert "Sunny." in msgs[1].content
            assert len(msgs) == 2  # user + assistant
