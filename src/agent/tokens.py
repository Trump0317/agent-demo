"""MVT 2.1 — Token 计数服务

封装 token 计数逻辑，优先使用 tiktoken，不可用时降级为字符估算。
"""

from __future__ import annotations

import logging

from src.agent.models import Message

logger = logging.getLogger(__name__)

# 尝试导入 tiktoken
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based estimation")


# 模型 → tiktoken 编码名映射
_MODEL_ENCODING_MAP: dict[str, str] = {
    "deepseek-chat": "cl100k_base",
    "deepseek-coder": "cl100k_base",
    "deepseek-reasoner": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def _get_encoding(model: str):
    """获取 tiktoken encoding，失败返回 None"""
    if not _TIKTOKEN_AVAILABLE:
        return None
    encoding_name = _MODEL_ENCODING_MAP.get(model, "cl100k_base")
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        logger.debug(f"Cannot get tiktoken encoding for {model}, falling back to estimation")
        return None


def count_tokens(messages: list[Message], model: str) -> int:
    """预估一组消息的 token 数

    优先使用 tiktoken 精确计数，不可用时降级为字符估算（~4 chars/token）。

    Args:
        messages: 消息列表
        model: 模型名（用于选择 tiktoken encoding）

    Returns:
        预估 token 数
    """
    enc = _get_encoding(model)

    if enc is not None:
        total = 0
        for msg in messages:
            total += _count_message_tokens_tiktoken(msg, enc)
        return total

    # 降级：简单字符估算
    total_chars = sum(len(msg.content) for msg in messages)
    # 加上 role 和结构开销（粗略 4 token/消息）
    overhead = len(messages) * 4
    return (total_chars // 4) + overhead


def _count_message_tokens_tiktoken(msg: Message, enc) -> int:
    """使用 tiktoken 计算单条消息的 token 数"""
    # 每条消息基础 token 开销（遵循 OpenAI 计数公式）
    tokens = 4  # 消息分隔符开销
    tokens += len(enc.encode(msg.role))
    tokens += len(enc.encode(msg.content))
    if msg.name:
        tokens += len(enc.encode(msg.name))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            tokens += len(enc.encode(tc.function.name))
            tokens += len(enc.encode(tc.function.arguments))
    return tokens


def estimate_tokens_from_text(text: str) -> int:
    """从纯文本估算 token 数（字符 / 4）

    Args:
        text: 文本内容

    Returns:
        估算 token 数
    """
    return max(1, len(text) // 4)
