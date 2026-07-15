# Agent Demo — 从零构建的最小可用 AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-145%20passed-green.svg)](.)

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

# 配置 API Key（三选一，优先级从高到低）
# 方式 1: 环境变量
export DEEPSEEK_API_KEY=sk-your-key

# 方式 2: .env 文件
echo "DEEPSEEK_API_KEY=sk-your-key" > .env

# 方式 3: config/config.yaml（已在 .gitignore 中，不会提交）
# 编辑 config/config.yaml，填入 llm.api_key
```

### 三种运行方式

**1. 交互式 CLI**（推荐日常使用）

```bash
# 一个终端 = 一个会话窗口
python examples/cli.py                  # 自动生成 session_id
python examples/cli.py --session work   # 固定窗口名，退出后重启可恢复历史
python examples/cli.py --session life -p "你是生活助理"

# CLI 内置命令
#   /exit   — 退出
#   /reset  — 清空当前窗口历史
#   /trace  — 查看 trace 文件路径
```

**2. 多窗口演示**（验证 Session 隔离）

```bash
python examples/multi_session_demo.py
# 两个 Agent 窗口独立交互，待办互不干扰，trace 保存在 ./traces/
```

**3. Python API**

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

### 交互式 CLI

```bash
# 终端 1 — 工作窗口
python examples/cli.py --session work

# 终端 2 — 生活窗口
python examples/cli.py --session life -p "你是生活助理，帮我管理待办"
```

```
╔══════════════════════════════════════════════════════╗
║                      Agent CLI                       ║
╠══════════════════════════════════════════════════════╣
║  session: work                                      ║
║  /exit  退出    /reset  清空历史          ║
║  /trace 查看 trace 文件路径               ║
╚══════════════════════════════════════════════════════╝

你> 帮我把提交周报、代码review加到待办
🤖 已添加: 1.提交周报  2.代码review
    [tokens: 850]

你> 代码review完成了吗？
🤖 还没有，需要我帮你标记为完成吗？

你> /trace
📄 ./traces/work.jsonl
   24 条记录, 8192 bytes

你> /exit
再见!
```

### 多窗口 Session 隔离

```bash
python examples/multi_session_demo.py
```

验证两个独立窗口的待办互不干扰、追问正确记忆、Trace 完整记录。

### Python API

```python
from src.agent.agent import Agent, BUILTIN_CALCULATOR

# 默认配置（内置全部工具）
agent = Agent()
response = await agent.run("Calculate 123 * 456")

# 精细控制
agent = Agent(
    system_prompt="You are a math tutor.",
    tools=[BUILTIN_CALCULATOR],
    compression="truncate",
    session_id="my-session",
)
```

参见 `examples/basic_usage.py`。

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

**当前测试统计**：145 个单元测试 + 8 个集成测试，全部通过。

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
│   ├── cli.py               # 交互式 CLI
│   ├── multi_session_demo.py # 多窗口 Session 隔离演示
│   └── basic_usage.py       # Python API 示例
├── reports/                 # Phase 完成报告
│   ├── phase1_report.md
│   ├── phase2_report.md
│   ├── phase3_report.md
│   └── phase4_report.md
├── PROBLEMS.md              # 问题与解决方案记录（21 个问题）
├── AI_PROMPT_LOG.md         # AI Prompt 与问题解决记录
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

- [AI_PROMPT_LOG.md](./AI_PROMPT_LOG.md) — AI Prompt 开发全过程
- [PROBLEMS.md](./PROBLEMS.md) — 21 个问题 + 解决方案（SDK 兼容、数据一致、测试、API 等）

---

## Trace 追踪

每次 Agent 运行自动在 `./traces/` 目录生成 JSONL 文件，每条记录包含 `trace_id, session_id, phase, timestamp, data`。

```bash
# 查看 trace 文件列表
ls ./traces/

# 查看某个 session 的全部 trace
cat traces/work.jsonl | python -m json.tool

# 按阶段筛选
grep '"llm_request"' traces/work.jsonl
grep '"tool_call"' traces/work.jsonl
grep '"llm_response"' traces/work.jsonl

# Phase 分布统计
grep -oP '"phase": "\K\w+' traces/work.jsonl | sort | uniq -c
```

## License

MIT
