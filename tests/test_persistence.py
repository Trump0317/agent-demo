"""Tests for MVT 2.4 — Session persistence."""
import os
import tempfile

import pytest

from src.agent.config import Config
from src.agent.exceptions import SessionNotFoundError, SessionSaveError
from src.agent.models import Message
from src.agent.persistence import JsonSessionStore, SessionStore
from src.agent.session import AgentSession


@pytest.fixture
def config():
    return Config(llm_api_key="sk-test")


@pytest.fixture
def store(config):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield JsonSessionStore(base_dir=tmpdir, config=config)


@pytest.fixture
def session(config):
    s = AgentSession(session_id="test-s1", system_prompt="You are helpful.", config=config)
    s._messages.append(Message(role="user", content="Hello"))
    s._recalc_tokens()
    return s


class TestJsonSessionStore:
    """JSON Session 存储测试"""

    @pytest.mark.asyncio
    async def test_save_and_file_exists(self, store, session):
        """save 后文件存在且格式正确"""
        await store.save(session)

        filepath = store._path(session.session_id)
        assert os.path.isfile(filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert data["session_id"] == "test-s1"
        assert "messages" in data
        assert "token_count" in data

    @pytest.mark.asyncio
    async def test_load_restores_session(self, store, session):
        """load 恢复的 session 与保存前一致"""
        await store.save(session)

        restored = await store.load("test-s1")
        assert restored.session_id == "test-s1"
        assert restored.token_count == session.token_count
        # 消息数应当匹配（注意 system prompt）
        assert len(restored.messages) == len(session.messages)
        assert restored.messages[-1].content == "Hello"

    @pytest.mark.asyncio
    async def test_list_sessions(self, store, session, config):
        """list_sessions 返回所有已保存 session 的 id"""
        s2 = AgentSession(session_id="test-s2", config=config)
        await store.save(session)
        await store.save(s2)

        ids = await store.list_sessions()
        assert "test-s1" in ids
        assert "test-s2" in ids
        assert len(ids) == 2

    @pytest.mark.asyncio
    async def test_load_nonexistent_raises(self, store):
        """load 不存在的 session 抛出 SessionNotFoundError"""
        with pytest.raises(SessionNotFoundError):
            await store.load("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self, store, session):
        """delete 删除 session"""
        await store.save(session)
        assert os.path.isfile(store._path("test-s1"))

        await store.delete("test-s1")
        assert not os.path.isfile(store._path("test-s1"))

    @pytest.mark.asyncio
    async def test_is_session_store(self, store):
        """JsonSessionStore 是 SessionStore 的子类"""
        assert isinstance(store, SessionStore)
