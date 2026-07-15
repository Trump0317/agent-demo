"""MVT 1.2 — 统一异常体系

所有异常类继承自 AgentError，按类别细分。
"""


class AgentError(Exception):
    """Agent 所有异常的基类"""
    pass


# ── 配置错误 ──

class ConfigError(AgentError):
    """配置错误（缺少 API Key 等）"""
    pass


# ── LLM 错误 ──

class LLMError(AgentError):
    """LLM 调用失败"""
    pass


class LLMTimeoutError(LLMError):
    """LLM 调用超时"""
    pass


class LLMAuthError(LLMError):
    """LLM 鉴权失败（401）"""
    pass


class LLMRateLimitError(LLMError):
    """LLM 限流（429）"""
    pass


# ── Tool 错误 ──

class ToolError(AgentError):
    """工具执行失败"""
    pass


class ToolNotFoundError(ToolError):
    """工具未找到"""
    pass


class ToolExecutionError(ToolError):
    """工具执行异常"""
    pass


# ── Session 错误 ──

class SessionError(AgentError):
    """Session 操作失败"""
    pass


class SessionNotFoundError(SessionError):
    """Session 未找到"""
    pass


class SessionSaveError(SessionError):
    """Session 保存失败"""
    pass


# ── Compression 错误 ──

class CompressionError(AgentError):
    """压缩失败"""
    pass


# ── Loop 错误 ──

class LoopError(AgentError):
    """Loop 异常（超过最大迭代等）"""
    pass
