"""Phase 4 — Agent 统一入口

封装 Config → DeepSeekClient → ToolRegistry → AgentSession → AgentLoop → TraceCollector
的完整组装链，对外暴露极简 API。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.agent.compression import CompressionStrategy, SummarizeStrategy, TruncateStrategy
from src.agent.config import Config
from src.agent.llm.deepseek import DeepSeekClient
from src.agent.loop import AgentLoop
from src.agent.models import LLMResponse, Message
from src.agent.persistence import JsonSessionStore, SessionStore
from src.agent.session import AgentSession
from src.agent.tools.base import Tool, ToolRegistry
from src.agent.tools.builtin.calculator import calculator
from src.agent.tools.builtin.search import search
from src.agent.tools.builtin.todo import todo
from src.agent.trace import JsonlTraceBackend, NoopTraceBackend, TraceCollector


# ── 内置工具 Schema 注册 ──

BUILTIN_CALCULATOR = Tool(
    name="calculator",
    description="Safely evaluate a mathematical expression. Supports +, -, *, /, **, //, %, ().",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The arithmetic expression to evaluate, e.g. '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
    handler=calculator,
)

BUILTIN_SEARCH = Tool(
    name="search",
    description="Search the web using DuckDuckGo. Returns a summary of results.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    },
    handler=search,
)

BUILTIN_TODO = Tool(
    name="todo",
    description=(
        "Manage a to-do list. Actions: add (with task description), "
        "list (show all), done (with task_id), clear (remove completed)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "One of: add, list, done, clear"},
            "task": {"type": "string", "description": "Task description (required for add)"},
            "task_id": {"type": "integer", "description": "Task ID (required for done)"},
            "session_id": {"type": "string", "description": "Session ID for todo isolation"},
        },
        "required": ["action"],
    },
    handler=todo,
)

ALL_BUILTIN_TOOLS = [BUILTIN_CALCULATOR, BUILTIN_SEARCH, BUILTIN_TODO]


def builtin_tools() -> ToolRegistry:
    """返回预装全部内置工具的 ToolRegistry 实例。"""
    return ToolRegistry(list(ALL_BUILTIN_TOOLS))


# ── Agent 统一入口 ──


@dataclass
class Agent:
    """AI Agent 统一入口 — 一行创建，一行运行。

    用法::

        agent = Agent()
        result = await agent.run("Calculate 1 + 1")

    也可精细控制::

        agent = Agent(
            system_prompt="You are a math tutor.",
            tools=[BUILTIN_CALCULATOR],
            compression="truncate",
            trace_enabled=True,
            session_id="resume-me",  # 恢复历史会话
        )
    """

    # ── 配置 ──
    config: Config | None = None
    """配置对象；若为 None 则从环境变量 / .env 自动加载"""

    # ── LLM ──
    model: str | None = None
    """模型名；若指定则覆盖 config.llm_model"""

    # ── System Prompt ──
    system_prompt: str = (
        "You are a helpful AI assistant with access to tools: "
        "calculator (math), search (web), and todo (task list). "
        "Use the appropriate tool when needed."
    )

    # ── 工具 ──
    tools: list[Tool] | Literal["builtin"] | ToolRegistry | None = "builtin"
    """工具配置:
    - ``"builtin"`` (默认): 预装全部 3 个内置工具
    - ``list[Tool]``: 自定义工具列表
    - ``ToolRegistry``: 直接传入 registry
    - ``None``: 无工具纯对话
    """

    # ── 会话 ──
    session_id: str | None = None
    """会话 ID；若指定且 session_store 有对应文件，则恢复历史会话"""

    # ── 压缩 ──
    compression: Literal["truncate", "summarize"] | None = None
    """上下文压缩策略:
    - ``"truncate"``: 滑动窗口截断
    - ``"summarize"``: LLM 摘要压缩
    - ``None``: 不压缩
    """

    # ── Trace ──
    trace_enabled: bool = True

    # ── 持久化 ──
    session_store: SessionStore | None = None
    """若提供，每次 run() 后自动保存 session"""

    # ── 内部状态 ──
    _initialized: bool = field(default=False, repr=False)
    _loop: AgentLoop | None = field(default=None, repr=False)
    _session: AgentSession | None = field(default=None, repr=False)
    _tools: ToolRegistry | None = field(default=None, repr=False)
    _trace: TraceCollector | None = field(default=None, repr=False)

    # ── 初始化 ──

    async def _init(self) -> None:
        """延迟初始化：在首次 run() 时自动调用"""
        if self._initialized:
            return

        # 1. Config
        if self.config is None:
            self.config = Config.from_env()

        if self.model:
            self.config.llm_model = self.model

        # 2. LLM Client
        llm = DeepSeekClient(self.config)

        # 3. Tools
        if self.tools == "builtin":
            self._tools = builtin_tools()
        elif isinstance(self.tools, ToolRegistry):
            self._tools = self.tools
        elif isinstance(self.tools, list):
            self._tools = ToolRegistry(self.tools)
        else:
            self._tools = ToolRegistry()

        # 3.5 Wrap todo tool with session_id（多窗口待办隔离）
        self._wrap_todo_session()

        # 4. Session
        if self.session_id and self.session_store:
            try:
                self._session = await self.session_store.load(self.session_id)
            except Exception:
                self._session = AgentSession(
                    session_id=self.session_id,
                    system_prompt=self.system_prompt,
                    config=self.config,
                )
        else:
            self._session = AgentSession(
                session_id=self.session_id,
                system_prompt=self.system_prompt,
                config=self.config,
            )

        # 5. Trace
        if self.trace_enabled:
            backend = JsonlTraceBackend(
                traces_dir=self.config.traces_dir,
                session_id=self._session.session_id,
            )
        else:
            backend = NoopTraceBackend()
        self._trace = TraceCollector(
            backend=backend,
            session_id=self._session.session_id,
        )

        # 6. Compression
        compression: CompressionStrategy | None = None
        if self.compression == "truncate":
            compression = TruncateStrategy()
        elif self.compression == "summarize":
            compression = SummarizeStrategy(llm)

        # 7. Loop
        self._loop = AgentLoop(
            llm_client=llm,
            tool_executor=self._tools,
            config=self.config,
            trace_collector=self._trace,
            session=self._session,
            compression_strategy=compression,
        )

        self._initialized = True

    def _wrap_todo_session(self) -> None:
        """为已注册的 todo 工具注入 session_id，实现多窗口待办隔离。

        若 todo 未注册则跳过。"""
        if self._tools is None:
            return
        try:
            todo_tool = self._tools.get("todo")
        except Exception:
            return

        original_handler = todo_tool.handler

        async def _todo_with_session(**kwargs):
            # 默认注入当前 Agent 的 session_id
            if "session_id" not in kwargs or not kwargs.get("session_id"):
                kwargs["session_id"] = self._session.session_id if self._session else "default"
            return await original_handler(**kwargs)

        todo_tool.handler = _todo_with_session

    # ── 公共 API ──

    async def run(self, task: str) -> LLMResponse:
        """执行一次 Agent 任务。

        Args:
            task: 用户输入的自然语言任务

        Returns:
            LLMResponse: LLM 的最终响应
        """
        await self._init()
        response = await self._loop.run(task)

        # 自动保存 session
        if self.session_store:
            await self.session_store.save(self._session)

        return response

    @property
    def messages(self) -> list[Message]:
        """当前会话的全部消息历史（只读副本）"""
        if self._session is None:
            return []
        return self._session.messages

    @property
    def token_count(self) -> int:
        """当前会话的 token 计数"""
        if self._session is None:
            return 0
        return self._session.token_count

    async def reset(self) -> None:
        """清空当前会话历史（保留 system_prompt）"""
        if self._session:
            await self._session.reset()

    async def add_tool(self, tool: Tool) -> None:
        """动态注册新工具"""
        await self._init()
        if self._tools is not None:
            self._tools.register(tool)
            # 如果是 todo 工具，自动注入 session_id
            if tool.name == "todo":
                self._wrap_todo_session()


# ── 便捷工厂函数 ──


def create_agent(
    *,
    system_prompt: str | None = None,
    tools: list[Tool] | Literal["builtin"] | None = "builtin",
    compression: Literal["truncate", "summarize"] | None = None,
    session_id: str | None = None,
    trace_enabled: bool = True,
    persist: bool = False,
    **kwargs,
) -> Agent:
    """一行创建 Agent 的工厂函数。

    Args:
        system_prompt: 系统提示词
        tools: 工具配置
        compression: 压缩策略
        session_id: 会话 ID
        trace_enabled: 是否启用 trace
        persist: 是否自动持久化 session

    Returns:
        已配置的 Agent 实例
    """
    store = None
    if persist:
        config = Config.from_env()
        store = JsonSessionStore(base_dir=config.sessions_dir, config=config)

    agent_kwargs: dict = {
        "tools": tools,
        "compression": compression,
        "session_id": session_id,
        "trace_enabled": trace_enabled,
        "session_store": store,
    }
    if system_prompt is not None:
        agent_kwargs["system_prompt"] = system_prompt
    agent_kwargs.update(kwargs)

    return Agent(**agent_kwargs)
