"""MVT 2.4 — Session 持久化

SessionStore (ABC): Session 持久化抽象。
JsonSessionStore: JSON 文件存储实现。
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod

from src.agent.config import Config
from src.agent.exceptions import SessionNotFoundError, SessionSaveError
from src.agent.session import AgentSession


class SessionStore(ABC):
    """Session 持久化抽象"""

    @abstractmethod
    async def save(self, session: AgentSession) -> None:
        """保存 Session"""
        ...

    @abstractmethod
    async def load(self, session_id: str) -> AgentSession:
        """加载 Session

        Raises:
            SessionNotFoundError: Session 不存在
        """
        ...

    @abstractmethod
    async def list_sessions(self) -> list[str]:
        """列出所有已保存的 Session ID"""
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """删除 Session"""
        ...


class JsonSessionStore(SessionStore):
    """JSON 文件存储：每个 session 一个 JSON 文件"""

    def __init__(self, base_dir: str, config: Config | None = None) -> None:
        self._base_dir = base_dir
        self._config = config

    def _path(self, session_id: str) -> str:
        return os.path.join(self._base_dir, f"{session_id}.json")

    async def save(self, session: AgentSession) -> None:
        """保存 Session 到 JSON 文件"""
        import asyncio

        try:
            os.makedirs(self._base_dir, exist_ok=True)
            data = session.to_dict()
            await asyncio.to_thread(self._write_json, self._path(session.session_id), data)
        except Exception as e:
            raise SessionSaveError(f"Failed to save session '{session.session_id}': {e}") from e

    async def load(self, session_id: str) -> AgentSession:
        """加载 Session

        Raises:
            SessionNotFoundError: Session 不存在
        """
        import asyncio

        path = self._path(session_id)
        if not os.path.isfile(path):
            raise SessionNotFoundError(f"Session '{session_id}' not found at {path}")

        try:
            data = await asyncio.to_thread(self._read_json, path)
            return AgentSession.from_dict(data, config=self._config)
        except SessionNotFoundError:
            raise
        except Exception as e:
            raise SessionSaveError(f"Failed to load session '{session_id}': {e}") from e

    async def list_sessions(self) -> list[str]:
        """列出所有已保存的 Session ID"""
        import asyncio

        try:
            files = await asyncio.to_thread(os.listdir, self._base_dir)
        except FileNotFoundError:
            return []
        return [
            f[:-5] for f in files
            if f.endswith(".json") and os.path.isfile(os.path.join(self._base_dir, f))
        ]

    async def delete(self, session_id: str) -> None:
        """删除 Session"""
        import asyncio

        path = self._path(session_id)
        if os.path.isfile(path):
            await asyncio.to_thread(os.remove, path)

    @staticmethod
    def _write_json(path: str, data: dict) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
