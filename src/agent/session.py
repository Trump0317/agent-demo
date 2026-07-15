"""MVT 2.2 — AgentSession

会话管理：持有消息历史、token 计数、system_prompt。
纯状态容器，不负责压缩决策。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.agent.config import Config
from src.agent.models import Message
from src.agent.tokens import count_tokens


class AgentSession:
    """对话会话

    管理消息历史、token 计数，提供消息增删方法。
    压缩编排由外部（AgentLoop）完成。
    """

    def __init__(
        self,
        session_id: str | None = None,
        system_prompt: str = "",
        config: Config | None = None,
    ) -> None:
        self._session_id = session_id or uuid4().hex
        self._config = config or Config()
        self._system_prompt = system_prompt
        self._messages: list[Message] = []
        self._token_count = 0
        self._created_at = datetime.now(timezone.utc)
        self._updated_at = self._created_at

        # 初始化 system prompt
        if system_prompt:
            self._messages.append(Message(role="system", content=system_prompt))
            self._recalc_tokens()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)  # 返回副本

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def config(self) -> Config:
        return self._config

    async def add_message(self, message: Message) -> None:
        """追加消息，更新 token 计数"""
        self._messages.append(message)
        self._recalc_tokens()
        self._updated_at = datetime.now(timezone.utc)

    async def add_user_task(self, task: str) -> None:
        """快捷方法：追加 user 消息"""
        await self.add_message(Message(role="user", content=task))

    def replace_messages(self, messages: list[Message]) -> None:
        """替换全部消息列表并重新计算 token 计数"""
        self._messages = list(messages)
        self._recalc_tokens()
        self._updated_at = datetime.now(timezone.utc)

    async def reset(self) -> None:
        """清空历史但保留 system_prompt"""
        if self._system_prompt:
            self._messages = [Message(role="system", content=self._system_prompt)]
        else:
            self._messages = []
        self._recalc_tokens()
        self._updated_at = datetime.now(timezone.utc)

    def set_token_count(self, count: int) -> None:
        """外部设置准确的 token 计数（如 LLM 返回的 usage）"""
        self._token_count = count

    def _recalc_tokens(self) -> None:
        """重新计算 token 计数"""
        self._token_count = count_tokens(self._messages, self._config.llm_model)

    def to_dict(self) -> dict:
        return {
            "session_id": self._session_id,
            "created_at": self._created_at.isoformat(),
            "updated_at": self._updated_at.isoformat(),
            "system_prompt": self._system_prompt,
            "messages": [m.to_dict() for m in self._messages],
            "token_count": self._token_count,
            "metadata": {},
        }

    @classmethod
    def from_dict(cls, d: dict, config: Config | None = None) -> "AgentSession":
        session = cls(
            session_id=d["session_id"],
            system_prompt=d.get("system_prompt", ""),
            config=config,
        )
        # 从 dict 恢复消息，跳过首条 system 消息（已在 __init__ 中添加）
        raw_messages: list[dict] = d.get("messages", [])
        filtered: list[dict] = []
        skip_system = True
        for m in raw_messages:
            if skip_system and m.get("role") == "system":
                skip_system = False
                continue  # 跳过，__init__ 已根据 system_prompt 添加
            filtered.append(m)

        session._messages = [Message.from_dict(m) for m in filtered]

        # 如果 system_prompt 不为空但 __init__ 已插入，确保不重复
        if session._system_prompt and (
            not session._messages or session._messages[0].role != "system"
        ):
            session._messages.insert(0, Message(role="system", content=session._system_prompt))

        session._recalc_tokens()
        session._created_at = datetime.fromisoformat(d["created_at"])
        session._updated_at = datetime.fromisoformat(d["updated_at"])
        return session
