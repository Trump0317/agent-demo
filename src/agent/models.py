"""MVT 1.1 — 核心数据模型

所有数据类定义：Message, ToolCall, ToolCallFunction, ToolResult, Usage,
LLMResponse, TraceRecord, TracePhase。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional


# ============================================================
# Message
# ============================================================


@dataclass
class Message:
    """对话消息"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: Optional[str] = None  # tool 角色时，关联的 tool_call id
    tool_calls: Optional[list[ToolCall]] = None  # assistant 角色时，ToolCall 列表
    name: Optional[str] = None  # tool 角色时，工具名称

    def to_openai_dict(self) -> dict:
        """转为 OpenAI API 兼容的字典格式"""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.name is not None:
            d["name"] = self.name
        return d

    def to_dict(self) -> dict:
        """序列化为普通 dict"""
        tc = None
        if self.tool_calls is not None:
            tc = [t.to_dict() for t in self.tool_calls]
        return {
            "role": self.role,
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "tool_calls": tc,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Message:
        """从 dict 反序列化"""
        tc = None
        if d.get("tool_calls"):
            tc = [ToolCall.from_dict(t) for t in d["tool_calls"]]
        return cls(
            role=d["role"],
            content=d["content"],
            tool_call_id=d.get("tool_call_id"),
            tool_calls=tc,
            name=d.get("name"),
        )


# ============================================================
# ToolCall / ToolCallFunction
# ============================================================


@dataclass
class ToolCallFunction:
    name: str
    arguments: str  # JSON 字符串

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, d: dict) -> ToolCallFunction:
        return cls(name=d["name"], arguments=d["arguments"])


@dataclass
class ToolCall:
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction = field(default_factory=lambda: ToolCallFunction("", ""))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": self.function.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolCall:
        return cls(
            id=d["id"],
            type=d.get("type", "function"),
            function=ToolCallFunction.from_dict(d["function"]),
        )


# ============================================================
# Usage
# ============================================================


@dataclass
class Usage:
    """LLM 调用 token 用量"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Usage:
        return cls(
            prompt_tokens=d.get("prompt_tokens", 0),
            completion_tokens=d.get("completion_tokens", 0),
            total_tokens=d.get("total_tokens", 0),
        )


# ============================================================
# LLMResponse
# ============================================================


@dataclass
class LLMResponse:
    """LLM 调用返回的统一响应"""

    content: Optional[str] = None  # 纯文本回复（无 tool_call 时有值）
    tool_calls: Optional[list[ToolCall]] = None  # 有工具调用时有值
    usage: Optional[Usage] = None
    finish_reason: Optional[str] = None  # "stop" / "tool_calls" / "length"
    model: Optional[str] = None

    @property
    def is_tool_call(self) -> bool:
        """是否包含工具调用请求"""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    @property
    def is_text_response(self) -> bool:
        """是否为纯文本回复"""
        return self.content is not None and not self.is_tool_call

    def to_dict(self) -> dict:
        tc = None
        if self.tool_calls is not None:
            tc = [t.to_dict() for t in self.tool_calls]
        return {
            "content": self.content,
            "tool_calls": tc,
            "usage": self.usage.to_dict() if self.usage else None,
            "finish_reason": self.finish_reason,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict) -> LLMResponse:
        tc = None
        if d.get("tool_calls"):
            tc = [ToolCall.from_dict(t) for t in d["tool_calls"]]
        return cls(
            content=d.get("content"),
            tool_calls=tc,
            usage=Usage.from_dict(d["usage"]) if d.get("usage") else None,
            finish_reason=d.get("finish_reason"),
            model=d.get("model"),
        )


# ============================================================
# ToolResult
# ============================================================


@dataclass
class ToolResult:
    """单次工具调用结果"""

    tool_call_id: str
    tool_name: str
    success: bool
    result: str  # 成功时为工具输出，失败时为错误消息
    latency_ms: float = 0.0

    def to_message(self) -> Message:
        """转为 tool 角色的 Message，供追加到对话历史"""
        return Message(
            role="tool",
            content=self.result,
            tool_call_id=self.tool_call_id,
            name=self.tool_name,
        )

    def to_dict(self) -> dict:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ToolResult:
        return cls(
            tool_call_id=d["tool_call_id"],
            tool_name=d["tool_name"],
            success=d["success"],
            result=d["result"],
            latency_ms=d.get("latency_ms", 0.0),
        )


# ============================================================
# TraceRecord / TracePhase
# ============================================================


class TracePhase(str, Enum):
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    LOOP_ITERATION = "loop_iteration"
    COMPRESSION = "compression"
    ERROR = "error"


@dataclass
class TraceRecord:
    trace_id: str
    session_id: str
    phase: TracePhase
    timestamp: str  # ISO 8601
    data: dict

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TraceRecord:
        return cls(
            trace_id=d["trace_id"],
            session_id=d["session_id"],
            phase=TracePhase(d["phase"]),
            timestamp=d["timestamp"],
            data=d["data"],
        )
