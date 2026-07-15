"""Tests for MVT 1.1 — Core Data Models."""
import pytest
from src.agent.models import (
    Message,
    ToolCall,
    ToolCallFunction,
    ToolResult,
    Usage,
    LLMResponse,
    TraceRecord,
    TracePhase,
)


class TestMessage:
    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_to_openai_dict_simple(self):
        msg = Message(role="system", content="You are helpful.")
        d = msg.to_openai_dict()
        assert d == {"role": "system", "content": "You are helpful."}

    def test_to_openai_dict_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            function=ToolCallFunction(name="calc", arguments='{"expr": "1+1"}'),
        )
        msg = Message(role="assistant", content="", tool_calls=[tc])
        d = msg.to_openai_dict()
        assert d["role"] == "assistant"
        assert d["tool_calls"][0]["id"] == "call_1"
        assert d["tool_calls"][0]["function"]["name"] == "calc"

    def test_to_openai_dict_tool_message(self):
        msg = Message(role="tool", content="42", tool_call_id="call_1", name="calc")
        d = msg.to_openai_dict()
        assert d["role"] == "tool"
        assert d["tool_call_id"] == "call_1"
        assert d["name"] == "calc"

    def test_roundtrip_serialization(self):
        """验证序列化/反序列化往返一致性"""
        original = Message(
            role="assistant",
            content="I'll calculate that",
            tool_calls=[
                ToolCall(
                    id="c1",
                    function=ToolCallFunction(name="math", arguments='{"x": 1}'),
                )
            ],
        )
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].id == "c1"


class TestLLMResponse:
    def test_text_response(self):
        r = LLMResponse(content="Hello!", finish_reason="stop", model="deepseek-chat")
        assert r.is_text_response
        assert not r.is_tool_call

    def test_tool_call_response(self):
        tc = ToolCall(
            id="c1",
            function=ToolCallFunction(name="search", arguments='{"q": "weather"}'),
        )
        r = LLMResponse(tool_calls=[tc], finish_reason="tool_calls", model="deepseek-chat")
        assert r.is_tool_call
        assert not r.is_text_response

    def test_roundtrip_serialization(self):
        original = LLMResponse(
            content="Hi",
            finish_reason="stop",
            model="gpt-4",
            usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        )
        d = original.to_dict()
        restored = LLMResponse.from_dict(d)
        assert restored.content == "Hi"
        assert restored.finish_reason == "stop"
        assert restored.usage.total_tokens == 12


class TestToolResult:
    def test_success_result(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", success=True, result="42")
        assert tr.success
        assert tr.result == "42"

    def test_to_message(self):
        tr = ToolResult(tool_call_id="c1", tool_name="calc", success=True, result="42")
        msg = tr.to_message()
        assert msg.role == "tool"
        assert msg.content == "42"
        assert msg.tool_call_id == "c1"
        assert msg.name == "calc"

    def test_failed_result_to_message(self):
        tr = ToolResult(
            tool_call_id="c2", tool_name="bad", success=False, result="Error: timeout"
        )
        msg = tr.to_message()
        assert msg.role == "tool"
        assert "Error" in msg.content


class TestTraceRecord:
    def test_trace_record_creation(self):
        tr = TraceRecord(
            trace_id="t1",
            session_id="s1",
            phase=TracePhase.LLM_REQUEST,
            timestamp="2026-01-01T00:00:00",
            data={"model": "deepseek-chat"},
        )
        d = tr.to_dict()
        assert d["phase"] == "llm_request"
        restored = TraceRecord.from_dict(d)
        assert restored.phase == TracePhase.LLM_REQUEST

    def test_all_phases(self):
        for phase in TracePhase:
            tr = TraceRecord(
                trace_id="t1", session_id="s1", phase=phase,
                timestamp="2026-01-01T00:00:00", data={},
            )
            assert tr.phase == phase
