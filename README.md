# Agent Demo — 从零构建的最小可用 AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-142%20passed-green.svg)](.)

纯 Python 实现，**零 Agent 框架依赖**（无 LangChain、AutoGPT、CrewAI 等），仅使用 `openai` SDK 调用 LLM API。适合学习 Agent 底层原理或作为自定义 Agent 项目的基础。

## 目录

1. [快速开始](#快速开始)
2. [系统设计](#系统设计)
3. [Session 与 Memory 机制](#session-与-memory-机制)
4. [Context 管理与压缩](#context-管理与压缩)
5. [工具系统](#工具系统)
6. [运行示例](#运行示例)
7. [测试](#测试)
8. [项目结构](#项目结构)
9. [开发记录](#开发记录)

---

## 快速开始

### 环境要求

- Python 3.10+
- DeepSeek API Key（或其他 OpenAI 兼容接口）

### 安装

```bash
# 克隆项目
git clone <repo-url> agent-demo
cd agent-demo

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e .

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY=sk-your-key
```

### 最简用法

```python
from src.agent.agent import Agent

agent = Agent()                                    # 一行创建（自动加载配置 + 预装工具）
result = await agent.run("Calculate 123 * 456")    # 一行运行
print(result.content)
```

---

## 系统设计

### 架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        Agent (统一入口)                           │
│  Config ← YAML / .env / os.environ                               │
│  DeepSeekClient ──→ LLM API (DeepSeek / OpenAI 兼容)             │
│  AgentSession ────→ 消息历史 + Token 计数                        │
│  ToolRegistry ────→ Calculator / Search / Todo / ...             │
│  TraceCollector ──→ JSONL Trace 文件                             │
│  CompressionStrategy → 上下文压缩                                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                     AgentLoop (核心循环)                           │
│                                                                   │
│  ┌──────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │ User │───→│ LLM Call │───→│ Tool     │───→│ LLM Call │───→ ...│
│  │ Input│    │          │    │ Executor │    │          │        │
│  └──────┘    └──────────┘    └──────────┘    └──────────┘        │
│                 │                                     │           │
│                 │ finish_reason="stop"                │           │
│                 └─────────────────────────────────────┘           │
│                              ▼                                    │
│                        Final Response                             │
└──────────────────────────────────────────────────────────────────┘
```

### 核心循环

1. **接收用户输入** — `user` 角色消息追加到 Session
2. **调用 LLM** — 传入完整对话历史 + 工具 Schema
3. **判断响应类型**：
   - `finish_reason="stop"` → 返回最终回答给用户
   - `finish_reason="tool_calls"` → 解析 ToolCall，进入步骤 4
4. **执行工具** — `ToolRegistry.execute()` 批量执行，返回 `list[ToolResult]`
5. **追加结果到上下文** — assistant 消息（含 tool_calls）+ tool 消息追加到 Session
6. **回到步骤 2** — 将工具结果反馈给 LLM，继续推理

### 核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据模型 | `models.py` | Message, ToolCall, LLMResponse, TraceRecord 等 |
| 配置管理 | `config.py` | YAML → .env → 环境变量 三级加载 |
| LLM 客户端 | `llm/deepseek.py` | DeepSeek API 调用，异常转换 |
| Agent Loop | `loop.py` | 核心循环编排 |
| Session | `session.py` | 消息历史、Token 计数 |
| 压缩策略 | `compression.py` | Truncate（滑动窗口）/ Summarize（LLM 摘要） |
| 持久化 | `persistence.py` | JSON 文件存储 Session |
| 工具系统 | `tools/base.py` | Tool + ToolRegistry |
| 内置工具 | `tools/builtin/` | calculator, search, todo |
| Trace | `trace.py` | JSONL 全链路追踪 |
| Agent 入口 | `agent.py` | 统一组装 + 延迟初始化 |

---

## Session 与 Memory 机制

### Session 生命周期

```
创建 Session → 累积消息 → 可选持久化 → 可选恢复 → 可选压缩 → 重置
```

### Memory 召回时机与放置方式

#### 1. 消息进入 Memory 的时机

| 消息类型 | 时机 | 触发者 |
|----------|------|--------|
| `system` | Session 创建时 | `Agent.__init__` |
| `user` | 每次用户输入 | `Agent.run(task)` → `session.add_user_task()` |
| `assistant`（文本） | LLM 返回 stop | `AgentLoop.run()` → `session.add_message()` |
| `assistant`（tool_calls） | LLM 返回 tool_calls | `AgentLoop.run()` → `session.add_message()` |
| `tool` | 工具执行完成后 | `AgentLoop.run()` → 追加至对话历史 |

#### 2. Memory 在 Context 中的放置

每次 LLM 调用传入**完整对话历史**（按时间顺序）：

```
messages = [
    Message(role="system",  content="You are a helpful assistant."),  # 始终第一条
    Message(role="user",    content="What's the weather?"),
    Message(role="assistant", content=None, tool_calls=[...]),         # tool_calls 请求
    Message(role="tool",    content="Sunny, 25°C", tool_call_id="..."),
    Message(role="assistant", content="The weather is sunny, 25°C."),
    Message(role="user",    content="Remind me to bring an umbrella"), # 新一轮追问
    ...
]
```

#### 3. 多 Session 隔离

每个 `AgentSession` 持有独立的 `_messages: list[Message]`，按 `session_id` 区分：

```python
# 窗口 1：查天气、记待办
agent1 = Agent(session_id="window-1", session_store=store)
await agent1.run("Check weather in Beijing")
await agent1.run("Add 'bring umbrella' to todo")

# 窗口 2：写周报、记待办
agent2 = Agent(session_id="window-2", session_store=store)
await agent2.run("Write weekly report summary")
await agent2.run("Add 'submit report' to todo")

# 窗口 1 继续对话——独立历史，互不影响
await agent1.run("What was the weather again?")
# Agent 记住之前的查天气和待办，不会看到窗口 2 的内容
```

#### 4. Memory 持久化与恢复

```python
# 保存（每次 run 后自动执行）
await store.save(session)

# 恢复（下次创建 Agent 时自动尝试）
agent = Agent(session_id="window-1", session_store=store)
# load 成功 → 续接历史对话
# load 失败 → 新 Session，不影响运行
```

---

## Context 管理与压缩

### Context 构成

每次 LLM 调用的 context 包含：

1. **system prompt** — 角色设定 + 工具使用说明
2. **历史消息** — 全部 user / assistant / tool 消息
3. **当前 user 输入** — 本次任务
4. **工具 Schema** — `tools` 参数（function calling 格式）

### 追问支持

#### 纯对话追问

```
User: "What is Python?"         → context: [system, user-1, assistant-1]
User: "What about its history?"  → context: [system, user-1, assistant-1, user-2]
```

LLM 从完整历史中理解"its"指 Python，无需额外处理。

#### 带工具的追问

```
User: "Calculate 100 + 200"     → context: [system, user-1, assistant(tool_call), tool(result), assistant-1]
User: "Now multiply that by 3"   → context: [system, ..., user-2]
```

LLM 从上下文中找到上次计算结果（300），自动继续推理。

### 压缩策略

当 `token_count / max_context_tokens > compression_threshold`（默认 0.8）时触发：

| 策略 | 实现 | 何时使用 |
|------|------|----------|
| **Truncate**（滑动窗口） | 保留 system_prompt + 最近 N 条消息 | 默认推荐，零额外 API 调用 |
| **Summarize**（LLM 摘要） | 调用 LLM 压缩早期消息为摘要 | 需保留更多历史信息时 |

---

## 工具系统

### 工具注册

```python
from src.agent.tools.base import Tool, ToolRegistry

async def my_tool(param: str) -> str:
    return f"Result: {param}"

registry = ToolRegistry()
registry.register(Tool(
    name="my_tool",
    description="Description for the LLM to understand when to use this tool.",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"},
        },
        "required": ["param"],
    },
    handler=my_tool,
))
```

### 内置工具

| 工具 | 功能 | 实现 |
|------|------|------|
| `calculator` | 安全算术求值 (+, -, *, /, **, %) | `ast.parse` + `NodeVisitor` |
| `search` | DuckDuckGo 搜索 | `urllib.request` + `asyncio.to_thread` |
| `todo` | 待办管理 (add/list/done/clear) | 内存 dict，线程安全，按 session 隔离 |

### LLM 决策流程

LLM 基于工具的 `name` + `description` + `parameters`（JSON Schema）自主决策：
- 需要计算 → 自动生成 `ToolCall(name="calculator", arguments={"expression": "1+2"})`
- 需要搜索 → 自动生成 `ToolCall(name="search", arguments={"query": "..."})`
- 直接回答 → `finish_reason="stop"` + `content="..."`

---

## 运行示例

```bash
# 设置 API Key
export DEEPSEEK_API_KEY=sk-your-key

# 运行交互示例
python examples/basic_usage.py
```

输出示例：
```
============================================================
Agent Demo
============================================================

📝 Task: Calculate 123 * 456 and tell me the result.
------------------------------------------------
🤖 Response: 123 × 456 = 56088
   Tokens: 145 (prompt=120, completion=25)

📝 Task: Add 'buy groceries' and 'call dentist' to my todo list...
------------------------------------------------
🤖 Response: I've added both tasks to your todo list...
   Tokens: 230 (prompt=180, completion=50)

============================================================
Session messages: 15
Session tokens:  850
```

### 高级用法

参见 `examples/basic_usage.py` 中的 `demo_advanced()` 函数：

```python
# 自定义工具集 + 压缩 + 持久化
agent = Agent(
    system_prompt="You are a math tutor.",
    tools=[BUILTIN_CALCULATOR],
    compression="truncate",
    session_id="my-session",
    session_store=store,
)
```

---

## 测试

```bash
# 单元测试（不需要 API Key）
pytest tests/ -v -m "not integration"

# 集成测试（需要 API Key + 网络）
pytest tests/ -v -m integration

# 全部测试
pytest tests/ -v

# 覆盖率
pytest tests/ -v --cov=src/agent --cov-report=term-missing
```

**当前测试统计**：142 个单元测试 + 9 个集成测试，全部通过。

---

## 项目结构

```
agent-demo/
├── config/
│   └── config.yaml          # YAML 配置文件
├── src/agent/
│   ├── agent.py             # Agent 统一入口（Phase 4）
│   ├── models.py            # 核心数据模型
│   ├── exceptions.py        # 统一异常体系（14 个类）
│   ├── config.py            # 配置管理（YAML/.env/env 三级加载）
│   ├── loop.py              # Agent 主循环
│   ├── session.py           # 会话管理
│   ├── compression.py       # 上下文压缩策略
│   ├── persistence.py       # Session JSON 持久化
│   ├── tokens.py            # Token 计数
│   ├── trace.py             # Trace 追踪系统
│   ├── llm/
│   │   ├── base.py          # LLMClient / ToolExecutor ABC
│   │   └── deepseek.py      # DeepSeek 客户端
│   └── tools/
│       ├── base.py          # Tool + ToolRegistry
│       └── builtin/
│           ├── calculator.py # 安全计算器
│           ├── search.py     # DuckDuckGo 搜索
│           └── todo.py       # 待办管理
├── tests/                   # 16 个测试文件
├── examples/
│   └── basic_usage.py       # 使用示例
├── reports/                 # Phase 完成报告
│   ├── phase1_report.md
│   ├── phase2_report.md
│   ├── phase3_report.md
│   └── phase4_report.md
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 开发记录

### 技术决策

1. **无框架依赖**：不使用 LangChain/AutoGPT/CrewAI，核心逻辑全自研
2. **接口优先**：所有核心模块基于 ABC 抽象接口，方便扩展
3. **异步优先**：基于 `asyncio`，IO 操作全异步
4. **三级配置**：YAML → .env → 环境变量，优先级逐级覆盖
5. **延迟初始化**：Agent 构造时不创建资源，首次 `run()` 才组装

### AI Prompt 与问题解决记录

详见 [AI_PROMPT_LOG.md](./AI_PROMPT_LOG.md)

---

## License

MIT
