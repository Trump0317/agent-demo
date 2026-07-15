"""Tests for MVT 2.3 / 3.4 — Compression strategies."""
import pytest

from src.agent.config import Config
from src.agent.compression import TruncateStrategy, SummarizeStrategy, CompressionStrategy
from src.agent.models import Message


@pytest.fixture
def config():
    return Config(
        llm_api_key="sk-test",
        max_context_tokens=64000,
        compression_threshold=0.8,
        compression_keep_recent=2,
    )


@pytest.fixture
def long_conversation():
    """生成一段较长的对话"""
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Tell me about Python."),
        Message(role="assistant", content="Python is a programming language."),
        Message(role="user", content="What about JavaScript?"),
        Message(role="assistant", content="JavaScript is for the web."),
        Message(role="user", content="Compare them."),
        Message(role="assistant", content="Key differences: syntax, ecosystem, etc."),
        Message(role="user", content="Tell me more about ecosystems."),
        Message(role="assistant", content="Python has PyPI, JavaScript has npm."),
    ]
    return messages


class TestTruncateStrategy:
    """Truncate 压缩策略测试"""

    @pytest.mark.asyncio
    async def test_system_prompt_preserved(self, config, long_conversation):
        """system prompt 始终被保留"""
        strategy = TruncateStrategy()
        result = await strategy.compress(long_conversation, max_tokens=5000, config=config)
        assert result[0].role == "system"
        assert result[0].content == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_recent_messages_preserved(self, config, long_conversation):
        """最近 N 条消息不被移除"""
        strategy = TruncateStrategy()
        result = await strategy.compress(long_conversation, max_tokens=100, config=config)

        # 保留最近的 keep_recent 条非 system 消息
        non_system = [m for m in result if m.role != "system"]
        assert len(non_system) >= config.compression_keep_recent
        # 最后一条应该是原始的 conversation 最后一条
        assert non_system[-1].content == long_conversation[-1].content

    @pytest.mark.asyncio
    async def test_token_count_below_threshold(self, config, long_conversation):
        """压缩后 token 数低于阈值"""
        strategy = TruncateStrategy()
        # 使用极低的 max_tokens（强制大量截断）
        result = await strategy.compress(long_conversation, max_tokens=50, config=config)

        from src.agent.tokens import count_tokens
        token_count = count_tokens(result, "deepseek-chat")
        threshold = int(50 * config.compression_threshold)
        assert token_count <= threshold, f"token_count={token_count} > threshold={threshold}"

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self, config, long_conversation):
        """不修改传入的原列表"""
        original_len = len(long_conversation)
        original_first = long_conversation[1].content  # 第二条消息
        strategy = TruncateStrategy()
        result = await strategy.compress(long_conversation, max_tokens=100, config=config)
        assert len(long_conversation) == original_len
        assert long_conversation[1].content == original_first

    @pytest.mark.asyncio
    async def test_small_conversation_unchanged(self, config):
        """不足 keep_recent 条消息时不应截断"""
        short = [
            Message(role="system", content="S"),
            Message(role="user", content="U"),
            Message(role="assistant", content="A"),
        ]
        strategy = TruncateStrategy()
        result = await strategy.compress(short, max_tokens=10, config=config)
        assert len(result) == len(short)


class TestSummarizeStrategy:
    """Summarize 压缩策略测试（Phase 3 MVT 3.4）"""

    @pytest.mark.asyncio
    async def test_messages_reduced(self, config, long_conversation):
        """压缩后消息数量减少"""
        from unittest.mock import AsyncMock
        from src.agent.models import LLMResponse

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = LLMResponse(
            content="Summary of the conversation about programming languages.",
            finish_reason="stop",
            model="mock",
        )

        strategy = SummarizeStrategy(mock_llm)
        result = await strategy.compress(long_conversation, max_tokens=100, config=config)

        # 结果应比原始少
        assert len(result) < len(long_conversation)
        # 应包含 system + summary + 最近 N 条
        assert result[0].role == "system"
        assert any("Summary" in m.content for m in result)

    @pytest.mark.asyncio
    async def test_recent_messages_unchanged(self, config, long_conversation):
        """最近 N 条消息内容不变"""
        from unittest.mock import AsyncMock
        from src.agent.models import LLMResponse

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = LLMResponse(
            content="Summary.",
            finish_reason="stop",
            model="mock",
        )

        strategy = SummarizeStrategy(mock_llm)
        result = await strategy.compress(long_conversation, max_tokens=100, config=config)

        # 最后 N 条非 system 消息应等于原始的最后 N 条
        expected_recent = [m for m in long_conversation if m.role != "system"][-config.compression_keep_recent:]
        actual_recent = [m for m in result if m.role != "system" and "Summary" not in m.content]
        for expected, actual in zip(expected_recent, actual_recent):
            assert actual.content == expected.content

    @pytest.mark.asyncio
    async def test_summary_message_is_system_role(self, config, long_conversation):
        """摘要消息 role 为 system"""
        from unittest.mock import AsyncMock
        from src.agent.models import LLMResponse

        mock_llm = AsyncMock()
        mock_llm.chat.return_value = LLMResponse(
            content="Summary.",
            finish_reason="stop",
            model="mock",
        )

        strategy = SummarizeStrategy(mock_llm)
        result = await strategy.compress(long_conversation, max_tokens=100, config=config)

        summary_msgs = [m for m in result if "Summary" in m.content]
        assert len(summary_msgs) > 0
        for msg in summary_msgs:
            assert msg.role == "system"

    @pytest.mark.asyncio
    async def test_is_compression_strategy(self):
        """SummarizeStrategy 是 CompressionStrategy 的子类"""
        from unittest.mock import AsyncMock
        strategy = SummarizeStrategy(AsyncMock())
        assert isinstance(strategy, CompressionStrategy)
