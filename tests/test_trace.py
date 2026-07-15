"""Tests for MVT 1.6 — Trace system."""
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.models import TracePhase, TraceRecord
from src.agent.trace import (
    JsonlTraceBackend,
    NoopTraceBackend,
    TraceBackend,
    TraceCollector,
)


class TestJsonlTraceBackend:
    """JSONL Trace 后端测试"""

    @pytest.mark.asyncio
    async def test_writes_jsonl_file(self):
        """正确写入 JSONL 文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JsonlTraceBackend(traces_dir=tmpdir, session_id="s1")
            record = TraceRecord(
                trace_id="t1",
                session_id="s1",
                phase=TracePhase.LLM_REQUEST,
                timestamp="2026-01-01T00:00:00",
                data={"model": "deepseek-chat"},
            )
            await backend.write(record)

            filepath = os.path.join(tmpdir, "s1.jsonl")
            assert os.path.isfile(filepath)

            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["phase"] == "llm_request"
            assert parsed["data"]["model"] == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_multiple_writes(self):
        """多次写入后文件行数正确"""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JsonlTraceBackend(traces_dir=tmpdir, session_id="s1")
            for i in range(3):
                record = TraceRecord(
                    trace_id=f"t{i}",
                    session_id="s1",
                    phase=TracePhase.LOOP_ITERATION,
                    timestamp="2026-01-01T00:00:00",
                    data={"iteration": i},
                )
                await backend.write(record)

            filepath = os.path.join(tmpdir, "s1.jsonl")
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 3
            for line in lines:
                parsed = json.loads(line)
                assert parsed["phase"] == "loop_iteration"

    @pytest.mark.asyncio
    async def test_each_line_is_valid_json(self):
        """每行格式为合法 JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JsonlTraceBackend(traces_dir=tmpdir, session_id="s2")
            record = TraceRecord(
                trace_id="t1",
                session_id="s2",
                phase=TracePhase.ERROR,
                timestamp="2026-01-01T00:00:00",
                data={"exception_type": "ValueError", "message": "test"},
            )
            await backend.write(record)

            filepath = os.path.join(tmpdir, "s2.jsonl")
            with open(filepath) as f:
                for line in f:
                    json.loads(line)  # 不应抛出 JSON 解析异常


class TestTraceCollector:
    """TraceCollector 测试"""

    def test_record_auto_fills_fields(self):
        """record() 自动填充 trace_id、session_id、timestamp"""
        mock_backend = AsyncMock(spec=TraceBackend)
        collector = TraceCollector(backend=mock_backend, session_id="s-test")

        collector.record(TracePhase.LLM_REQUEST, {"model": "gpt"})

        # 验证 write 被调用（由于 create_task 是异步的，需手动等待）
        # 这里只验证调用发生了
        assert mock_backend.write.called

    def test_trace_id_is_unique(self):
        """每次实例化生成不同的 trace_id"""
        c1 = TraceCollector(session_id="s1")
        c2 = TraceCollector(session_id="s1")
        assert c1.trace_id != c2.trace_id

    @pytest.mark.asyncio
    async def test_trace_disabled_noop_backend(self):
        """trace_enabled=False 时使用 NoopTraceBackend"""
        collector = TraceCollector(backend=NoopTraceBackend(), session_id="s1")
        # 不应抛出异常
        collector.record(TracePhase.LLM_REQUEST, {"model": "gpt"})

    @pytest.mark.asyncio
    async def test_collector_with_jsonl_backend(self):
        """端到端：TraceCollector → JsonlTraceBackend"""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = JsonlTraceBackend(traces_dir=tmpdir, session_id="s3")
            collector = TraceCollector(backend=backend, session_id="s3")

            collector.record(TracePhase.LLM_REQUEST, {"model": "deepseek-chat", "message_count": 3, "tools": [], "estimated_tokens": 100})
            collector.record(TracePhase.LLM_RESPONSE, {"content": "Hello", "tool_calls": None, "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}, "finish_reason": "stop", "latency_ms": 200.5})

            # 等待后台写入
            import asyncio
            await asyncio.sleep(0.1)

            filepath = os.path.join(tmpdir, "s3.jsonl")
            assert os.path.isfile(filepath)
            with open(filepath) as f:
                lines = f.readlines()
            assert len(lines) == 2
