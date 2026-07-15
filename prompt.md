# Agent 开发指导 Prompt

## 项目概述

使用 Python 从零构建一个最小可用的 AI Agent，无框架依赖（不使用 LangChain、AutoGPT 等）。允许的第三方依赖仅限：`openai`（LLM SDK）、`pyyaml`（YAML 配置文件解析）、`tiktoken`（Token 计数）、`pytest`（测试）。

---

## 一、总体架构约束

1. **零框架依赖**：核心逻辑不引入任何 Agent 框架。允许的第三方依赖：
   - `openai` — OpenAI Python SDK，用于调用 LLM API（OpenAI 兼容接口均可），同时提供准确的 token 用量（`response.usage.prompt_tokens` / `completion_tokens`）
   - `pytest` — 测试框架（仅开发依赖）
   - 其余一律使用 Python 标准库（`asyncio`、`json`、`logging`、`urllib`、`dataclasses`、`abc` 等），HTTP 请求用标准库 `urllib.request` 配合 `asyncio.to_thread`

2. **接口优先**：所有核心模块均基于抽象接口（ABC），方便后续扩展。接口定义后，Phase 之间的实现替换不应破坏已有代码。

3. **异步优先**：核心 loop 基于 `asyncio`，所有 IO 操作（LLM 调用、文件读写）均异步执行。

4. **可观测性**：全链路 Trace 记录，每个关键阶段均持久化。

---

## 二、核心数据模型（所有 Phase 共享，Phase 1 最先实现）

> 以下模型为全项目统一规范，后续所有模块引用这些模型，不再各自定义。

### 2.1 Message

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class Message:
    """对话消息"""
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: Optional[str] = None       # tool 角色时，关联的 tool_call id
    tool_calls: Optional[list[ToolCall]] = None  # assistant 角色时，ToolCall 列表
    name: Optional[str] = None               # tool 角色时，工具名称

    def to_openai_dict(self) -> dict:
        """转为 OpenAI API 兼容的字典格式"""
        ...
```

### 2.2 ToolCall / ToolCallFunction

```python
@dataclass
class ToolCallFunction:
    name: str
    arguments: str  # JSON 字符串

@dataclass
class ToolCall:
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction = field(default_factory=ToolCallFunction)
```

### 2.3 Usage

```python
@dataclass
class Usage:
    """LLM 调用 token 用量"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
```

### 2.4 LLMResponse

```python
@dataclass
class LLMResponse:
    """LLM 调用返回的统一响应"""
    content: Optional[str] = None          # 纯文本回复（无 tool_call 时有值）
    tool_calls: Optional[list[ToolCall]] = None  # 有工具调用时有值
    usage: Optional[Usage] = None          # token 用量
    finish_reason: Optional[str] = None    # "stop" / "tool_calls" / "length"
    model: Optional[str] = None            # 实际使用的模型名
```

### 2.5 ToolResult

```python
@dataclass
class ToolResult:
    """单次工具调用结果"""
    tool_call_id: str
    tool_name: str
    success: bool
    result: str          # 成功时为工具输出，失败时为错误消息
    latency_ms: float = 0

    def to_message(self) -> Message:
        """转为 tool 角色的 Message，供追加到对话历史"""
        return Message(
            role="tool",
            content=self.result,
            tool_call_id=self.tool_call_id,
            name=self.tool_name,
        )
```

### 2.6 工具定义 Schema（JSON Schema 格式）

```python
# Tool 定义中 parameters 字段遵循 JSON Schema 规范，示例：
{
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "算术表达式，如 '2 + 3 * 4'"
        }
    },
    "required": ["expression"]
}
```

### 2.7 TraceRecord

```python
from enum import Enum

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
    timestamp: str   # ISO 8601
    data: dict
```

**Trace data 按 phase 的字段规范**：

| phase | data 字段 |
|-------|----------|
| `llm_request` | `model`, `message_count`, `tools`(名称列表), `estimated_tokens` |
| `llm_response` | `content`(截取前200字符), `tool_calls`(名称列表), `usage`({prompt,completion,total}), `finish_reason`, `latency_ms` |
| `tool_call` | `tool_name`, `arguments` |
| `tool_result` | `tool_name`, `success`, `result`(截取前500字符), `latency_ms` |
| `loop_iteration` | `iteration`, `max_iterations`, `total_tokens_so_far` |
| `compression` | `strategy`, `before_tokens`, `after_tokens`, `removed_count` |
| `error` | `exception_type`, `message`, `phase`(出错阶段), `traceback` |

---

## 三、统一异常体系（Phase 1 建立）

```
AgentError (base)
├── ConfigError          # 配置错误（缺少 API Key 等）
├── LLMError             # LLM 调用失败
│   ├── LLMTimeoutError  # 超时
│   ├── LLMAuthError     # 鉴权失败
│   └── LLMRateLimitError# 限流
├── ToolError            # 工具执行失败
│   ├── ToolNotFoundError
│   └── ToolExecutionError
├── SessionError         # Session 操作失败
│   ├── SessionNotFoundError
│   └── SessionSaveError
├── CompressionError     # 压缩失败
└── LoopError            # Loop 异常（超过最大迭代等）
```

---

## 四、配置管理（Phase 1 建立）

### 4.1 Config 数据类

```python
from dataclasses import dataclass, field

@dataclass
class Config:
    # ── LLM ──
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    llm_timeout: int = 60          # 单次 LLM 调用超时（秒）
    llm_max_retries: int = 2       # LLM 调用失败重试次数

    # ── Agent Loop ──
    max_iterations: int = 10
    max_context_tokens: int = 64000

    # ── Tool ──
    tool_timeout: int = 30         # 单次工具调用超时（秒）

    # ── Paths ──
    sessions_dir: str = "./sessions"
    traces_dir: str = "./traces"

    # ── Trace ──
    trace_enabled: bool = True
    trace_content_max_length: int = 500  # trace 中截断内容的最大长度

    # ── Compression ──
    compression_threshold: float = 0.8   # token 使用率达到此比例触发压缩
    compression_keep_recent: int = 5     # 压缩时保留最近 N 条消息

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Config":
        """从环境变量和 .env 文件加载配置"""
        ...
```

### 4.2 环境变量映射

| 配置项 | 环境变量 | 默认值 |
|--------|---------|--------|
| `llm_api_key` | `DEEPSEEK_API_KEY` | （必填，无默认值） |
| `llm_base_url` | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `llm_model` | `DEEPSEEK_MODEL` | `deepseek-chat` |
| `llm_timeout` | `AGENT_LLM_TIMEOUT` | `60` |
| `llm_max_retries` | `AGENT_LLM_MAX_RETRIES` | `2` |
| `max_iterations` | `AGENT_MAX_ITERATIONS` | `10` |
| `max_context_tokens` | `AGENT_MAX_CONTEXT_TOKENS` | `64000` |
| `tool_timeout` | `AGENT_TOOL_TIMEOUT` | `30` |
| `sessions_dir` | `AGENT_SESSIONS_DIR` | `./sessions` |
| `traces_dir` | `AGENT_TRACES_DIR` | `./traces` |
| `trace_enabled` | `AGENT_TRACE_ENABLED` | `true` |
| `trace_content_max_length` | `AGENT_TRACE_CONTENT_MAX_LENGTH` | `500` |
| `compression_threshold` | `AGENT_COMPRESSION_THRESHOLD` | `0.8` |
| `compression_keep_recent` | `AGENT_COMPRESSION_KEEP_RECENT` | `5` |

### 4.3 YAML 配置文件（`config/config.yaml`）

> Phase 4 新增：支持从 YAML 文件加载配置，优先级低于 .env 和环境变量。

```yaml
# Agent Configuration
# 优先级：环境变量 > .env 文件 > config.yaml > 默认值

llm:
  api_key: ""                # DEEPSEEK_API_KEY（必填，建议通过环境变量设置）
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  timeout: 60
  max_retries: 2

loop:
  max_iterations: 10
  max_context_tokens: 64000

tool:
  timeout: 30

paths:
  sessions_dir: "./sessions"
  traces_dir: "./traces"

trace:
  enabled: true
  content_max_length: 500

compression:
  threshold: 0.8
  keep_recent: 5
```

**Config 新增方法**：

```python
@classmethod
def from_yaml(cls, path: str = "config/config.yaml") -> "Config":
    """从 YAML 文件加载配置"""
    ...

@classmethod
def from_env(
    cls,
    env_file: str | None = ".env",
    yaml_file: str | None = "config/config.yaml",
) -> "Config":
    """三级优先级加载：环境变量 > .env > config.yaml > 默认值"""
    ...
```

### 4.4 `.env.example` 内容

```bash
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
AGENT_MAX_ITERATIONS=10
AGENT_MAX_CONTEXT_TOKENS=64000
AGENT_LLM_TIMEOUT=60
AGENT_LLM_MAX_RETRIES=2
AGENT_SESSIONS_DIR=./sessions
AGENT_TRACES_DIR=./traces
AGENT_TRACE_ENABLED=true
```

---

## 五、抽象接口一览（分属各 Phase，此处汇总）

| 接口 | 所属 Phase | 职责 |
|------|-----------|------|
| `LLMClient` (ABC) | Phase 1 | LLM 调用抽象，`chat(messages, tools) -> LLMResponse` |
| `ToolExecutor` (ABC) | Phase 1（定义+Stub） | 工具执行抽象，`execute(tool_calls) -> list[ToolResult]`。Phase 1 用 Stub 占位，Phase 3 由 `ToolRegistry` 实现 |
| `TraceBackend` (ABC) | Phase 1 | Trace 写入后端，`write(record: TraceRecord) -> None` |
| `SessionStore` (ABC) | Phase 2 | Session 持久化，`save/load/list/delete` |
| `CompressionStrategy` (ABC) | Phase 2（接口+Truncate）/ Phase 3（Summarize） | 压缩策略，`compress(messages, max_tokens, config) -> list[Message]` |

---

## 六、分阶段目标

### Phase 1: Foundation + Agent Loop

> **定位**：建立全部数据模型、配置系统、异常体系、LLM 客户端和最小可用的 Agent 主循环。这是后续所有 Phase 的基础。

#### MVT 1.1 — 核心数据模型

**产出**：`src/agent/models.py`

实现 `Message`、`ToolCall`、`ToolCallFunction`、`ToolResult`、`Usage`、`LLMResponse`、`TraceRecord`、`TracePhase` 等所有数据类。每个数据类需包含 `to_dict()` 序列化方法。

**验收标准**：
- [ ] 所有数据类可正确实例化
- [ ] `Message.to_openai_dict()` 输出符合 OpenAI API 格式
- [ ] `LLMResponse` 可正确表示"纯文本回复"和"Tool Call 请求"两种场景
- [ ] 至少 3 个数据模型相关单测

#### MVT 1.2 — 统一异常体系

**产出**：`src/agent/exceptions.py`

按第三节定义实现完整异常树。每个异常类携带清晰的错误消息。

**验收标准**：
- [ ] 所有异常类可实例化，继承链正确
- [ ] `isinstance(exc, AgentError)` 对子类返回 True
- [ ] 至少 2 个异常相关单测

#### MVT 1.3 — 配置管理

**产出**：`src/agent/config.py`

实现 `Config` 数据类，`from_env()` 方法从环境变量和 `.env` 文件加载。若 `DEEPSEEK_API_KEY` 缺失，抛出 `ConfigError`。

**验收标准**：
- [ ] 从环境变量正确加载所有配置项
- [ ] `.env` 文件缺失时不报错（使用默认值）
- [ ] `DEEPSEEK_API_KEY` 缺失时正确抛出 `ConfigError`
- [ ] `from_env()` 支持传入自定义 `.env` 路径
- [ ] 至少 4 个配置相关单测

#### MVT 1.4 — LLM 抽象接口 + DeepSeek 实现

**产出**：`src/agent/llm/base.py`、`src/agent/llm/deepseek.py`

- `LLMClient`（ABC）：定义 `async chat(messages: list[Message], tools: list[dict] | None = None) -> LLMResponse`
- `DeepSeekClient`：基于 `openai.AsyncOpenAI` 实现，base_url 指向 `https://api.deepseek.com/v1`，返回标准 `LLMResponse`
- 错误处理：捕获 OpenAI SDK 异常，转换为第二节定义的 `LLMError` 子类（超时→`LLMTimeoutError`，401→`LLMAuthError`，429→`LLMRateLimitError`）

**验收标准**：
- [ ] `LLMClient` 接口定义完整，可被继承
- [ ] 向 DeepSeek 发送一条简单消息并收到 `LLMResponse`（集成测试）
- [ ] `LLMResponse.usage` 包含准确的 `prompt_tokens` 和 `completion_tokens`
- [ ] 鉴权失败时抛出 `LLMAuthError`（Mock 401 响应）
- [ ] 超时时抛出 `LLMTimeoutError`（Mock 超时）
- [ ] 至少 5 个单测 + 1 个集成测试

#### MVT 1.5 — Agent Loop

**产出**：`src/agent/loop.py`

```python
class AgentLoop:
    def __init__(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor | None = None,  # Phase 1 为 Stub，Phase 3 换为 ToolRegistry
        config: Config | None = None,
        trace_collector: TraceCollector | None = None,
    ): ...

    async def run(self, task: str) -> LLMResponse:
        """执行一次 Agent Loop，返回最终 LLM 响应"""
        ...
```

Loop 逻辑：
1. 接收 `task`，构造初始消息列表 `[Message(role="user", content=task)]`
2. 调用 `llm_client.chat(messages, tools)`（tools 从 `tool_executor` 获取 schema 列表）
3. 根据 `LLMResponse.finish_reason` 判断：
   - `"stop"` → 返回 `LLMResponse`
   - `"tool_calls"` → 调用 `tool_executor.execute(tool_calls)`，将结果转为 `Message(role="tool")` 追加到消息历史，回到步骤 2
4. 超过 `config.max_iterations` → 抛出 `LoopError`
5. 每次迭代记录 Trace

**验收标准**：
- [ ] Mock LLM 返回 `finish_reason="stop"`：Loop 正常终止并返回响应
- [ ] Mock LLM 先返回 tool_call 再返回 stop：Loop 正确执行工具并迭代
- [ ] 超过 `max_iterations` 时抛出 `LoopError`
- [ ] 无工具时的纯对话场景正常
- [ ] Trace 记录中包含每次迭代的 `loop_iteration` 事件
- [ ] 至少 6 个单测

#### MVT 1.6 — Trace 系统

**产出**：`src/agent/trace.py`

- `TraceBackend`（ABC）：`async write(record: TraceRecord) -> None`
- `JsonlTraceBackend`：写入 `{traces_dir}/{session_id}.jsonl`，每条 TraceRecord 为一行 JSON
- `TraceCollector`：聚合 trace_id、session_id，提供 `record(phase, data)` 便捷方法，内部通过 `datetime` 生成 ISO 8601 时间戳和 `uuid4` 生成 trace_id

**验收标准**：
- [ ] `JsonlTraceBackend` 正确写入 JSONL 文件
- [ ] 多次写入后文件行数正确，每行格式合法 JSON
- [ ] `TraceCollector.record()` 自动填充 trace_id、session_id、timestamp
- [ ] `trace_enabled=False` 时 `TraceCollector` 不写入（用 Mock Backend 验证）
- [ ] 至少 4 个单测

#### Phase 1 完成后产出文件

```
src/agent/
├── __init__.py
├── models.py          # MVT 1.1
├── exceptions.py      # MVT 1.2
├── config.py          # MVT 1.3
├── loop.py            # MVT 1.5
├── trace.py           # MVT 1.6
└── llm/
    ├── __init__.py
    ├── base.py        # MVT 1.4
    └── deepseek.py    # MVT 1.4
tests/
├── conftest.py
├── test_models.py
├── test_exceptions.py
├── test_config.py
├── test_llm.py
├── test_loop.py
├── test_trace.py
└── test_integration.py
```

---

### Phase 2: Session Management

> **定位**：在 Phase 1 基础上，加入会话管理能力。Session 管理对话历史、token 计数、上下文窗口控制和 JSON 持久化。
> 
> **前置依赖**：Phase 1 的 `Message`、`Config`、`AgentError`、`LLMClient` 接口必须稳定。

#### MVT 2.1 — Token 计数服务

**产出**：`src/agent/tokens.py`

- 封装 `tiktoken`（`pip install tiktoken`，或使用 `openai` SDK 内置的 token 计数）对外暴露统一接口
- 提供 `count_tokens(messages: list[Message], model: str) -> int`：预估一组消息的 token 数
- 若 `tiktoken` 不可用，降级为简单字符估算并 log 警告
- 每次 LLM 调用后，从 `LLMResponse.usage.total_tokens` 累加获得准确值

**验收标准**：
- [ ] `count_tokens` 返回合理数字（与 API 实际返回 ±10% 即可）
- [ ] `tiktoken` 不可用时正确降级
- [ ] 至少 3 个单测

#### MVT 2.2 — AgentSession

**产出**：`src/agent/session.py`

```python
class AgentSession:
    def __init__(
        self,
        session_id: str | None,
        system_prompt: str = "",
        config: Config | None = None,
    ): ...

    @property
    def messages(self) -> list[Message]: ...
    @property
    def token_count(self) -> int: ...

    async def add_message(self, message: Message) -> None:
        """追加消息，更新 token 计数（基于预估）"""
        ...

    async def add_user_task(self, task: str) -> None:
        """快捷方法：追加 user 消息"""
        ...

    def replace_messages(self, messages: list[Message]) -> None:
        """替换全部消息列表并重新计算 token 计数。供外部压缩后调用。"""
        ...

    async def reset(self) -> None:
        """清空历史但保留 system_prompt"""
        ...

    # session_id 在 __init__ 时若为 None 则自动生成 uuid4
```

**核心行为**：
1. `add_message()` 后自动累加 token 计数（使用 MVT 2.1 的 `count_tokens` 预估）
2. `replace_messages()` 完全替换消息列表并重算 token 计数
3. Session 不持有压缩策略引用，压缩由外部（AgentLoop）编排
4. `reset()` 保留 `system_prompt`，清空其余消息，重置 token 计数

> **设计决策**：Session 是纯状态容器，不负责压缩决策。压缩编排由 AgentLoop 在 MVT 2.5 中完成。

**验收标准**：
- [ ] 添加消息后 `token_count` 正确变化
- [ ] `system_prompt` 作为首条消息正确管理
- [ ] `replace_messages()` 替换消息后 token 计数正确重算
- [ ] `reset()` 后只剩 system_prompt，token 计数正确
- [ ] 至少 5 个单测

#### MVT 2.3 — 上下文压缩（Truncate 策略）

**产出**：`src/agent/compression.py`

```python
class CompressionStrategy(ABC):
    @abstractmethod
    async def compress(
        self, messages: list[Message], max_tokens: int, config: Config
    ) -> list[Message]: ...

class TruncateStrategy(CompressionStrategy):
    """滑动窗口：保留 system_prompt + 最近 N 条消息，丢弃中间最旧的消息"""
    ...
```

**具体行为**：
- 始终保留 `system` 角色消息
- 从最早的非 system 消息开始移除，直到 token 数低于 `max_tokens * compression_threshold`
- 至少保留 `config.compression_keep_recent` 条最近消息
- 返回压缩后的消息列表（不修改原列表）

**验收标准**：
- [ ] system prompt 始终被保留
- [ ] 压缩后 token 数低于阈值
- [ ] 最近 N 条消息不被移除
- [ ] 不修改传入的原列表
- [ ] 至少 4 个单测

#### MVT 2.4 — Session 持久化

**产出**：`src/agent/persistence.py`

```python
class SessionStore(ABC):
    @abstractmethod
    async def save(self, session: AgentSession) -> None: ...
    @abstractmethod
    async def load(self, session_id: str) -> AgentSession: ...
    @abstractmethod
    async def list_sessions(self) -> list[str]: ...
    @abstractmethod
    async def delete(self, session_id: str) -> None: ...

class JsonSessionStore(SessionStore):
    """JSON 文件存储，每个 session 一个 JSON 文件"""
    def __init__(self, base_dir: str): ...
```

**JSON 文件格式**：
```json
{
  "session_id": "uuid",
  "created_at": "2026-07-15T12:00:00",
  "updated_at": "2026-07-15T12:05:00",
  "system_prompt": "...",
  "messages": [ {...}, {...} ],
  "token_count": 1234,
  "metadata": {}
}
```

**验收标准**：
- [ ] `save` 后文件存在且格式正确
- [ ] `load` 恢复的 session 与保存前一致（消息、token_count、system_prompt）
- [ ] `list_sessions` 返回所有已保存 session 的 id
- [ ] `load` 不存在的 session 抛出 `SessionNotFoundError`
- [ ] `save` 失败抛出 `SessionSaveError`
- [ ] 至少 5 个单测

#### MVT 2.5 — Session 与 AgentLoop 集成

**产出**：更新 `src/agent/loop.py`

在 `AgentLoop` 中集成 `AgentSession` 和 `CompressionStrategy`：

- `AgentLoop.__init__` 新增参数：`session: AgentSession | None`、`compression_strategy: CompressionStrategy | None`
- `run()` 使用 session 的消息历史（而非临时列表）
- 每次调用 LLM **前**：检查 `session.token_count / config.max_context_tokens > compression_threshold`，若超阈值则调用 `compression_strategy.compress()`，然后用 `session.replace_messages()` 更新
- 每次 LLM 调用**后**：用 `LLMResponse.usage` 修正 session 的 token 计数（将准确值写入 session）
- 每次迭代结束后自动 `session_store.save(session)`
- 若提供 `session_id`，从 `SessionStore` 加载已有 session

> **设计要点**：压缩编排在 Loop 中完成，Session 不持有压缩策略引用。Loop 是唯一的编排者。
> Loop 的 `tool_executor` 参数不变——Phase 3 的 `ToolRegistry` 继承 `ToolExecutor`，无需修改 Loop。

**验收标准**：
- [ ] 使用已有 session 时可恢复对话历史
- [ ] 多轮对话场景下自动压缩触发且不影响对话连贯性（集成测试）
- [ ] 至少 3 个单测 + 1 个集成测试

#### Phase 2 完成后新增文件

```
src/agent/
├── tokens.py          # MVT 2.1
├── session.py         # MVT 2.2
├── compression.py     # MVT 2.3
└── persistence.py     # MVT 2.4
tests/
├── test_tokens.py
├── test_session.py
├── test_compression.py
└── test_persistence.py
```

---

### Phase 4: Agent 组装层（统一入口）

> **定位**：封装 Config → DeepSeekClient → ToolRegistry → AgentSession → AgentLoop → TraceCollector 的完整组装链，对外暴露极简 API。

#### MVT 4.1 — Agent 统一入口

**产出**：`src/agent/agent.py`

```python
from dataclasses import dataclass, field

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
            session_id="resume-me",
        )
    """

    config: Config | None = None
    model: str | None = None
    system_prompt: str = "..."
    tools: list[Tool] | Literal["builtin"] | ToolRegistry | None = "builtin"
    session_id: str | None = None
    compression: Literal["truncate", "summarize"] | None = None
    trace_enabled: bool = True
    session_store: SessionStore | None = None

    async def run(self, task: str) -> LLMResponse:
        """执行一次 Agent 任务（首次调用自动延迟初始化）"""
        ...

    @property
    def messages(self) -> list[Message]: ...

    @property
    def token_count(self) -> int: ...

    async def reset(self) -> None: ...

    async def add_tool(self, tool: Tool) -> None: ...
```

**核心设计**：
1. **延迟初始化**：构造时不创建资源，首次 `run()` 才组装全套组件
2. **工具自动加载**：`tools="builtin"` 默认预装 calculator/search/todo
3. **Session 自动恢复**：传入 `session_id` + `session_store` 自动从文件恢复
4. **自动持久化**：`session_store` 非 None 时，每次 `run()` 后自动保存

**便捷工厂函数**：

```python
def create_agent(
    *,
    system_prompt: str | None = None,
    tools: list[Tool] | Literal["builtin"] | None = "builtin",
    compression: Literal["truncate", "summarize"] | None = None,
    session_id: str | None = None,
    trace_enabled: bool = True,
    persist: bool = False,
) -> Agent:
    """一行创建 Agent 的工厂函数"""
    ...
```

**验收标准**：
- [ ] Agent 默认构造一行代码完成，无需手动组装
- [ ] 首次 `run()` 自动延迟初始化，后续 `run()` 不重复创建
- [ ] 内置工具自动注册并传入 LLM
- [ ] `session_store` 提供时自动持久化
- [ ] `create_agent()` 工厂函数正确创建 Agent
- [ ] 至少 15 个单测

#### Phase 4 完成后新增/更新文件

```
src/agent/
├── agent.py           # MVT 4.1 Agent 统一入口
config/
├── config.yaml        # YAML 配置文件
tests/
├── test_agent.py      # Agent 入口测试
```

---

### Phase 3: Tool System

> **定位**：在 Phase 1/2 的稳定接口之上，实现完整的 Tool 注册/调用机制，用真正的 `ToolRegistry` 替换 Phase 1 的 Stub。
>
> **前置依赖**：`ToolExecutor` ABC（Phase 1 已定义）、`LLMResponse.tool_calls`、`Message`、`ToolResult`（Phase 1 已定义）。`ToolRegistry` 继承 `ToolExecutor`，无需修改 `AgentLoop`。

#### MVT 3.1 — Tool 基座

**产出**：`src/agent/tools/base.py`

```python
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Any

class ToolExecutor(ABC):
    """工具执行抽象。Phase 1 用 Stub 占位，Phase 3 由 ToolRegistry 实现。"""
    @abstractmethod
    async def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        ...

    @abstractmethod
    def list_schemas(self) -> list[dict]:
        """列出所有工具的 OpenAI Schema，供 LLM 调用时传入"""
        ...


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict              # JSON Schema
    handler: Callable[..., Awaitable[Any]]  # 异步函数

    def to_openai_schema(self) -> dict:
        """转为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


class ToolRegistry(ToolExecutor):
    def __init__(self, tools: list[Tool] | None = None): ...

    def register(self, tool: Tool) -> None:
        """注册工具，同名覆盖"""
        ...

    def get(self, name: str) -> Tool:
        """获取工具，不存在抛出 ToolNotFoundError"""
        ...

    def list_schemas(self) -> list[dict]:
        """列出所有工具的 OpenAI Schema"""
        ...

    async def execute(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
        """批量执行工具调用，返回 ToolResult 列表。
        单个工具失败不影响其他工具，失败的 ToolResult.success=False。
        超时控制：使用 asyncio.wait_for，超时抛出 ToolExecutionError。
        """
        ...
```

**验收标准**：
- [ ] `register` 注册工具后可正确 `get` 获取
- [ ] `get` 不存在的工具抛出 `ToolNotFoundError`
- [ ] `list_schemas` 输出符合 OpenAI function calling 格式
- [ ] `execute` 正确调用工具并返回 `list[ToolResult]`
- [ ] 工具执行超时抛出 `ToolExecutionError`
- [ ] `ToolRegistry` 继承 `ToolExecutor`，可无缝替换 Phase 1 的 Stub
- [ ] 至少 5 个单测

#### MVT 3.2 — 内置工具（calculator, search, todo）

**产出**：
- `src/agent/tools/builtin/calculator.py`
- `src/agent/tools/builtin/search.py`
- `src/agent/tools/builtin/todo.py`

**calculator**：
```python
# 使用 ast.parse + ast.NodeVisitor 安全求值
# 支持：+, -, *, /, **, (), 整数和浮点数
# 禁止：任何函数调用、属性访问、导入语句
async def calculator(expression: str) -> str:
    """安全计算算术表达式，返回计算结果字符串"""
```

**search**：
```python
# 使用 urllib.request + asyncio.to_thread 调用 DuckDuckGo Instant Answer API
# URL: https://api.duckduckgo.com/?q={query}&format=json&no_html=1
# 返回 AbstractText + RelatedTopics 摘要
async def search(query: str) -> str:
    """搜索查询，返回摘要结果"""
```

**todo**：
```python
# 内存存储（dict[str, list[dict]]），按 session 隔离
# 支持 action: "add" / "list" / "done" / "clear"
async def todo(action: str, task: str = "", task_id: int = 0) -> str:
    """
    管理待办事项。
    action='add' + task='事项描述' → 添加
    action='list' → 列出所有
    action='done' + task_id=N → 标记完成
    action='clear' → 清空已完成项
    """
```

**验收标准**：
- [ ] calculator：正常表达式返回正确结果，恶意输入（`__import__`、`eval` 等）被拦截
- [ ] search：返回非空字符串结果
- [ ] todo：add/list/done/clear 行为正确，状态一致
- [ ] 每个工具提供至少 3 个单测
- [ ] 至少 9 个工具单测总计

#### MVT 3.3 — 端到端集成验证

**产出**：无需修改 `loop.py`（`ToolRegistry` 继承 `ToolExecutor`，与 Phase 1 接口兼容）

**验证内容**：
- 构造完整 Agent：`DeepSeekClient` + `ToolRegistry`(含 3 个内置工具) + `AgentSession` + `JsonSessionStore`
- 端到端测试：LLM 请求 calculator → Agent 返回正确计算结果
- 端到端测试：LLM 请求 search → Agent 返回搜索结果
- 端到端测试：LLM 请求 todo add + list → Agent 正确管理待办
- 工具执行超时时 Loop 不崩溃，错误信息传给 LLM 继续处理
- 多工具连续调用场景（如先 search 再 calculator）

> **关键**：此 MVT 不修改 `loop.py`。`ToolRegistry` 继承 Phase 1 定义的 `ToolExecutor` ABC，直接传入 `AgentLoop(tool_executor=registry)`。验证的是集成正确性，而非新功能。

**验收标准**：
- [ ] `ToolRegistry` 实例可直接作为 `AgentLoop(tool_executor=registry)` 的参数
- [ ] 端到端：LLM 请求 calculator → 正确计算结果
- [ ] 端到端：LLM 请求 search → 返回搜索结果
- [ ] 端到端：LLM 请求 todo add + list → 正确管理待办
- [ ] 工具执行超时时 Loop 不崩溃，错误信息传给 LLM
- [ ] 至少 3 个集成测试（需要 API Key）

#### MVT 3.4 — Summarize 压缩策略

**产出**：更新 `src/agent/compression.py`

```python
class SummarizeStrategy(CompressionStrategy):
    """LLM 摘要压缩：调用 LLM 对早期消息做摘要"""
    def __init__(self, llm_client: LLMClient): ...

    async def compress(
        self, messages: list[Message], max_tokens: int, config: Config
    ) -> list[Message]:
        # 1. 选取保留范围外的消息（保留最近 N 条）
        # 2. 调用 LLM 生成摘要
        # 3. 构造摘要 Message(role="system", content="[历史摘要]...")
        # 4. 返回：[system_prompt] + [摘要] + [最近 N 条]
        ...
```

**验收标准**：
- [ ] 压缩后消息数量减少
- [ ] 保留最近 N 条消息完整不变
- [ ] 摘要消息 role 为 system
- [ ] 压缩后 token 数减少（需集成测试验证，使用真实 LLM）
- [ ] 至少 3 个单测 + 1 个集成测试

#### Phase 3 完成后新增文件

```
src/agent/tools/
├── __init__.py
├── base.py             # MVT 3.1
└── builtin/
    ├── __init__.py
    ├── calculator.py    # MVT 3.2
    ├── search.py        # MVT 3.2
    └── todo.py          # MVT 3.2
tests/
├── test_tools_base.py
├── test_tools_builtin.py
└── test_integration.py  # 随 Phase 逐步丰富
```

---

## 七、反馈机制

### 7.1 自动化测试（pytest）

**要求**：
- 每个 MVT 至少达到规定的测试数量且全部通过
- 测试覆盖：正常路径、异常路径、边界条件、Mock 场景
- Mock LLM 响应用于确定性测试（构造固定 `LLMResponse`，不依赖真实 API）
- 集成测试（文件 `test_integration.py`，使用 `@pytest.mark.integration` 标记）仅在 `DEEPSEEK_API_KEY` 环境变量存在时运行

**运行方式**：
```bash
# 运行所有单元测试（不依赖外部 API）
pytest tests/ -v -m "not integration"

# 运行集成测试（需要 API Key）
pytest tests/ -v -m integration

# 运行全部测试
pytest tests/ -v

# 带覆盖率报告
pytest tests/ -v --cov=src/agent --cov-report=term-missing
```

**测试文件与 MVT 对应**：

| 测试文件 | 对应 MVT | Phase |
|---------|---------|-------|
| `test_models.py` | MVT 1.1 | 1 |
| `test_exceptions.py` | MVT 1.2 | 1 |
| `test_config.py` | MVT 1.3 | 1 |
| `test_llm.py` | MVT 1.4 | 1 |
| `test_loop.py` | MVT 1.5, 2.5 | 1,2 |
| `test_trace.py` | MVT 1.6 | 1 |
| `test_tokens.py` | MVT 2.1 | 2 |
| `test_session.py` | MVT 2.2 | 2 |
| `test_compression.py` | MVT 2.3, 3.4 | 2,3 |
| `test_persistence.py` | MVT 2.4 | 2 |
| `test_tools_base.py` | MVT 3.1 | 3 |
| `test_tools_builtin.py` | MVT 3.2 | 3 |
| `test_agent.py` | MVT 4.1 | 4 |
| `test_integration.py` | 跨 Phase 集成 + MVT 3.3 | 1,2,3 |

### 7.2 全链路 Trace 追踪

已在 Phase 1 MVT 1.6 中定义。关键设计决策：
- `TraceBackend` 为抽象接口，默认实现 `JsonlTraceBackend`
- `TraceCollector` 是 Agent Loop 使用 Trace 的唯一入口
- Trace 写入失败不抛异常、不中断主流程，仅 log 错误

### 7.3 代码质量检查（建议，非强制）

```bash
# Lint & 格式化
pip install ruff
ruff check src/ tests/
ruff format src/ tests/

# 类型检查（逐步启用）
pip install mypy
mypy src/
```

---

## 八、观察与迭代规则

### 8.1 开发流程

1. **按 Phase → MVT 顺序推进**：必须完成当前 MVT 的全部验收标准并通过测试，才能进入下一 MVT。同一 Phase 内的 MVT 也按序执行。
2. **MVT 产出**：代码 + 测试 + 测试通过截图/输出。
3. **Phase 产出**：Phase 内所有 MVT 完成后，生成 Phase 完成报告。

### 8.2 Phase 完成报告模板

每个 Phase 完成后，生成 `reports/phase{N}_report.md`：

```markdown
# Phase {N} 完成报告

## 完成日期
YYYY-MM-DD

## MVT 完成情况
| MVT | 名称 | 状态 | 测试数 |
|-----|------|------|--------|
| 1.1 | 核心数据模型 | ✅ | 3 |
| 1.2 | 统一异常体系 | ✅ | 2 |
| ... | ...  | ... | ...  |

## 实现摘要
简要描述本 Phase 做了什么。

## 文件清单
- src/agent/xxx.py — 说明
- tests/test_xxx.py — 说明

## 测试执行输出
\`\`\`
pytest tests/ -v -m "not integration"
... (粘贴实际输出)
\`\`\`

## 已知问题 / 遗留项
- 无  /  列出问题

## 对外暴露的稳定接口（供后续 Phase 使用）
- `Message` / `LLMResponse` / `Usage` — 全局数据模型
- `Config.from_env()` — 配置入口
- `LLMClient.chat()` — LLM 调用
- `AgentLoop.run()` — 主循环
- ...
```

### 8.3 错误处理迭代规则

1. **语法/导入错误**：立即修复，确保代码可运行。
2. **测试失败**：
   - 分析失败原因，判断是实现错误还是测试本身的问题
   - 最多重试 3 次修复同一测试
   - 3 次后仍失败：记录到报告"已知问题"，标记为需人工介入
3. **接口变更**：若发现已有接口需要修改（非新增方法），先评估对已完成 MVT 的影响，获得确认后再修改。优先通过新增接口方法或新策略类来扩展。
4. **API 错误**（DeepSeek 返回异常）：统一转换为异常体系的对应类型，记录 Trace，不静默失败。

### 8.4 代码迭代原则

- 每次修改后运行**相关测试文件**，确保不引入回归
- 数据模型（`models.py`）变更需运行全量测试
- 新增功能优先通过扩展接口（新增 Strategy / Backend 实现类），而非修改已有代码
- 每个 MVT 的代码可独立运行和测试，不依赖后续 MVT

---

## 九、目录结构总览（最终）

```
agent-demo/
├── .env.example
├── .gitignore
├── pyproject.toml
├── prompt.md                 # 本文件
├── README.md                 # 项目文档
├── AI_PROMPT_LOG.md          # AI Prompt 与问题解决记录
├── config/
│   └── config.yaml           # YAML 配置文件（Phase 4）
├── reports/
│   ├── phase1_report.md
│   ├── phase2_report.md
│   ├── phase3_report.md
│   └── phase4_report.md
├── traces/                   # .gitignore
├── sessions/                 # .gitignore
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── models.py         # MVT 1.1 全部数据模型
│       ├── exceptions.py     # MVT 1.2 统一异常体系
│       ├── config.py         # MVT 1.3 + Phase 4：YAML/.env/env 三级加载
│       ├── loop.py           # MVT 1.5 + 2.5 AgentLoop
│       ├── trace.py          # MVT 1.6 Trace 系统
│       ├── tokens.py         # MVT 2.1 Token 计数
│       ├── session.py        # MVT 2.2 AgentSession
│       ├── compression.py    # MVT 2.3 + 3.4 压缩策略
│       ├── persistence.py    # MVT 2.4 Session 持久化
│       ├── agent.py          # MVT 4.1 Agent 统一入口
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── base.py       # MVT 1.4 LLMClient + ToolExecutor ABC
│       │   └── deepseek.py   # MVT 1.4 DeepSeek 实现
│       └── tools/
│           ├── __init__.py
│           ├── base.py       # MVT 3.1 Tool + ToolRegistry
│           └── builtin/
│               ├── __init__.py
│               ├── calculator.py  # MVT 3.2
│               ├── search.py      # MVT 3.2
│               └── todo.py        # MVT 3.2
├── tests/
│   ├── __init__.py
│   ├── test_models.py       # MVT 1.1
│   ├── test_exceptions.py   # MVT 1.2
│   ├── test_config.py       # MVT 1.3 + Phase 4 YAML
│   ├── test_llm.py          # MVT 1.4
│   ├── test_loop.py         # MVT 1.5, 2.5
│   ├── test_trace.py        # MVT 1.6
│   ├── test_tokens.py       # MVT 2.1
│   ├── test_session.py      # MVT 2.2 + 多窗口隔离
│   ├── test_compression.py  # MVT 2.3, 3.4
│   ├── test_persistence.py  # MVT 2.4
│   ├── test_tools_base.py   # MVT 3.1
│   ├── test_tools_builtin.py # MVT 3.2
│   ├── test_agent.py        # MVT 4.1
│   └── test_integration.py  # 跨 Phase 集成
└── examples/
    └── basic_usage.py
```
