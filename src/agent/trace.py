"""MVT 1.6 — Trace 系统

TraceBackend (ABC): Trace 写入后端抽象。
JsonlTraceBackend: JSONL 文件后端。
TraceCollector: 聚合 trace_id/session_id，提供便捷 record 方法。
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import uuid4

from src.agent.models import TracePhase, TraceRecord

logger = logging.getLogger(__name__)


class TraceBackend(ABC):
    """Trace 写入后端抽象"""

    @abstractmethod
    async def write(self, record: TraceRecord) -> None:
        """写入一条 Trace 记录"""
        ...


class JsonlTraceBackend(TraceBackend):
    """JSONL 文件后端：每条 TraceRecord 写入一行 JSON"""

    def __init__(self, traces_dir: str, session_id: str) -> None:
        self._path = os.path.join(traces_dir, f"{session_id}.jsonl")

    async def write(self, record: TraceRecord) -> None:
        """写入一条 Trace 记录（追加到 JSONL 文件）"""
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            # 使用 asyncio.to_thread 异步写入
            import asyncio
            await asyncio.to_thread(self._write_sync, record)
        except Exception as e:
            logger.error(f"Failed to write trace record: {e}")

    def _write_sync(self, record: TraceRecord) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


class NoopTraceBackend(TraceBackend):
    """空后端：不写入任何 trace（trace_enabled=False 时使用）"""

    async def write(self, record: TraceRecord) -> None:
        pass


class TraceCollector:
    """Trace 收集器

    聚合 trace_id、session_id，提供便捷的 record 方法。
    """

    def __init__(
        self,
        backend: TraceBackend | None = None,
        session_id: str | None = None,
    ) -> None:
        self._backend = backend or NoopTraceBackend()
        self._trace_id = uuid4().hex
        self._session_id = session_id or "unknown"

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def session_id(self) -> str:
        return self._session_id

    def record(self, phase: TracePhase, data: dict) -> None:
        """记录一条 Trace

        Args:
            phase: Trace 阶段
            data: 阶段数据（字段参见 TraceRecord data 规范）
        """
        record = TraceRecord(
            trace_id=self._trace_id,
            session_id=self._session_id,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
        )
        # 非阻塞：schedule 写入 task，不等待完成
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._backend.write(record))
        except RuntimeError:
            # 没有 running loop，同步写入
            import asyncio as _asyncio
            _asyncio.run(self._backend.write(record))
