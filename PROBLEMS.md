# 问题与解决方案记录

## 项目概述

从零构建 AI Agent，4 个 Phase、16 个模块、145 个测试。以下是开发过程中遇到的全部问题及解决方案。

---

## Phase 1: Foundation + Agent Loop

### 问题 1.1 — OpenAI SDK v2.x 异常初始化失败

**现象**：Mock `AuthenticationError("401")` 抛 `TypeError: APIStatusError.__init__() missing 2 required keyword-only arguments: 'response' and 'body'`

**根因**：OpenAI v2.x SDK 的异常类构造函数必须传入合法的 `httpx.Response` 对象。

**解决方案**：

```python
def _make_mock_response(status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)

AuthenticationError("msg", response=_make_mock_response(401), body=None)
```

**影响范围**：`tests/test_llm.py` — `test_auth_error`、`test_rate_limit_error`

---

### 问题 1.2 — `RateLimitError` 同样需要 httpx.Response

**现象**：同问题 1.1，`RateLimitError("429", response=None, body=None)` 报 `AttributeError: 'NoneType' object has no attribute 'request'`

**根因**：`RateLimitError.__init__` 内部调用 `response.request`，传入 `None` 会崩溃。

**解决方案**：使用相同的 `_make_mock_response(429)` 辅助函数。

---

### 问题 1.3 — ToolExecutor 与 ToolRegistry 的接口兼容性

**设计问题**：Phase 1 定义了 `ToolExecutor` ABC（带 `_StubToolExecutor`），Phase 3 的 `ToolRegistry` 需要无缝替换它。

**解决方案**：`ToolRegistry` 直接继承 `ToolExecutor` ABC，实现 `execute()` 和 `list_schemas()`。`AgentLoop.__init__` 接受 `ToolExecutor | None`，默认用 Stub。

**关键设计**：接口优先。Phase 3 的 `AgentLoop(tool_executor=registry)` 无需修改 Loop 代码。

---

### 问题 1.4 — TraceCollector 与 AgentLoop 的循环导入

**现象**：`loop.py` import `trace.py`，`trace.py` 可能依赖 `loop.py`。

**解决方案**：使用 `TYPE_CHECKING` 延迟导入。

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.trace import TraceCollector

class AgentLoop:
    def __init__(self, trace_collector: "TraceCollector | None" = None):
        ...
```

---

### 问题 1.5 — `asyncio.iscoroutinefunction` 在 Python 3.14 中废弃

**现象**：`DeprecationWarning: 'asyncio.iscoroutinefunction' is deprecated`

**解决方案**：改用 `inspect.iscoroutinefunction()`。

---

## Phase 2: Session Management

### 问题 2.1 — tiktoken 不可用时的降级

**现象**：`import tiktoken` 失败导致 `count_tokens` 崩溃。

**解决方案**：

```python
try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based estimation")

def count_tokens(messages, model):
    if _TIKTOKEN_AVAILABLE:
        return _count_with_tiktoken(...)
    else:
        return sum(len(m.content) for m in messages) // 4 + overhead
```

---

### 问题 2.2 — Session 恢复时 system prompt 重复

**现象**：`AgentSession.__init__` 根据 `system_prompt` 自动插入 system 消息；`from_dict()` 恢复 JSON 中的消息列表时，再次匹配到 `system_prompt`，导致 system 消息重复。

**解决方案**：`from_dict()` 从 JSON 恢复消息时，跳过首条 `role="system"` 消息（由 `__init__` 自动添加）。

```python
skip_system = True
for m in raw_messages:
    if skip_system and m.get("role") == "system":
        skip_system = False
        continue  # 跳过，__init__ 已根据 system_prompt 添加
    filtered.append(m)
```

---

### 问题 2.3 — Token 计数的预估与精确双重策略

**设计问题**：`count_tokens()` 是预估值，`response.usage.total_tokens` 是精确值。用哪个？

**解决方案**：双路径：

| 场景 | 使用值 | 原因 |
|------|--------|------|
| 压缩判断（LLM 调用前） | `session.token_count`（预估） | 速度快，无需 API 调用 |
| LLM 调用后 | `session.set_token_count(usage.total_tokens)`（精确） | 覆盖为准确值 |
| 状态展示 | 精确值 | 面向用户 |

---

## Phase 3: Tool System

### 问题 3.1 — Calculator 安全求值：拦截恶意输入

**现象**：如何安全求值 `"1+1"` 但阻止 `"__import__('os').system('rm -rf /')"`？

**解决方案**：`ast.parse(mode="eval")` + 自定义 `NodeVisitor` 白名单。

```python
_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ...}
_SAFE_CONSTANTS = (int, float)

class _CalcVisitor(ast.NodeVisitor):
    def generic_visit(self, node):
        raise ValueError(f"Unsupported expression: {type(node).__name__}")
```

任何不在白名单的 AST 节点（`Call`、`Attribute`、`Import` 等）都会被 `generic_visit()` 拒绝。

**测试覆盖**：`test_blocks_import`、`test_blocks_eval`、`test_blocks_attribute_access`

---

### 问题 3.2 — search 工具：同步 HTTP 请求在异步环境中

**设计约束**：Prompt 要求使用标准库 `urllib.request`（同步），但 Agent 全链路异步。

**解决方案**：`asyncio.to_thread()` 包装。

```python
response_text = await asyncio.to_thread(_fetch_url, url, timeout=15)
```

---

### 问题 3.3 — Todo 工具：多 Session 隔离 + 线程安全

**设计问题**：Todo 是内存存储，多个 Agent 实例并发调用。

**解决方案**：

```python
_storage: dict[str, list[dict]] = {}  # session_id → todos
_lock = threading.Lock()

def _get_todos(session_id: str) -> list[dict]:
    with _lock:
        if session_id not in _storage:
            _storage[session_id] = []
        return _storage[session_id]
```

**后续改进**（Phase 4）：`Agent._wrap_todo_session()` 自动注入 `session_id`，LLM 无需手动传递。

---

### 问题 3.4 — ToolRegistry.execute() 异常变量未绑定

**现象**：`except ToolNotFoundError: f"Error: {e}"` → `UnboundLocalError: cannot access local variable 'e'`

**根因**：忘记 `except ToolNotFoundError as e:` 中的 `as e`。

**解决方案**：补上 `as e`。

---

## Phase 4: Agent 组装层

### 问题 4.1 — `add_tool` 后工具未生效

**现象**：先 `agent.add_tool(new_tool)` 后 `agent.run("task")`，LLM 调用时 tools 参数不包含新工具。

**根因**：`AgentLoop` 在初始化时缓存了 `tools_schemas`。

**解决方案**：`run()` 每次从 `ToolRegistry.list_schemas()` 实时获取，确保动态注册即时生效。

---

### 问题 4.2 — YAML 配置加载与 .env 优先级顺序

**设计问题**：YAML、.env、环境变量如何排优先级？

**解决方案**：三级加载链

```
环境变量 (最高)
  ↓ 覆盖
.env 文件
  ↓ 覆盖
config/config.yaml
  ↓ 覆盖
默认值 (最低)
```

`from_env()` 实现：

```python
# 层 1: YAML 文件 → base config
base = cls.from_yaml(yaml_file)

# 层 2: .env 文件 → 写入 environ (不覆盖已有)
cls._load_env_file(env_file)

# 层 3: 环境变量 → 以 YAML 值为 fallback 读取
return cls(
    llm_model=os.getenv("DEEPSEEK_MODEL", base.llm_model),
    ...
)
```

---

### 问题 4.3 — 多 Agent 共享同一个 Tool 对象导致 handler 互相覆盖

**现象**：两个 Agent 窗口的 todo 混在一起。窗口 1 问"有没有周报？"返回了窗口 2 的数据。

**根因**：`Agent` 类使用模块级 `ALL_BUILTIN_TOOLS` 列表创建 ToolRegistry。两个 Agent 共享同一个 `BUILTIN_TODO` 对象。`_wrap_todo_session()` 修改 `todo_tool.handler` 时，第二个 Agent 覆盖了第一个的 wrapper。

```
Agent1._wrap() → BUILTIN_TODO.handler = wrapper(session_id="window-1")
Agent2._wrap() → BUILTIN_TODO.handler = wrapper(session_id="window-2")  ← 覆盖!
Agent1.run()  → 实际执行的是 session_id="window-2" 的 wrapper
```

**解决方案**：`builtin_tools()` 使用 `copy.deepcopy` 创建全新 Tool 实例。

```python
def builtin_tools() -> ToolRegistry:
    import copy
    return ToolRegistry([copy.deepcopy(t) for t in ALL_BUILTIN_TOOLS])
```

同时，用户传入的自定义 Tool 列表也做深拷贝：

```python
elif isinstance(self.tools, list):
    self._tools = ToolRegistry([copy.deepcopy(t) for t in self.tools])
```

**教训**：共享可变对象 + 运行时修改属性 = 隐蔽 bug。多实例场景下要么深拷贝、要么工厂函数每次创建新对象。

---

### 问题 4.4 — 集成测试 fixture 只检查环境变量，不检查 YAML

**现象**：API key 写在 `config/config.yaml` 中，但 `pytest -m integration` 全部 SKIP。

**根因**：`api_config` fixture 直接 `os.getenv("DEEPSEEK_API_KEY")` 检查，不走 `Config.from_env()` 的三级加载。

**解决方案**：fixture 改用 `Config.from_env()` 加载，只在 `ConfigError` 时才 skip。

```python
# 之前
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    pytest.skip(...)

# 之后
try:
    return Config.from_env()
except ConfigError:
    pytest.skip(...)
```

---

## 测试相关

### 问题 T.1 — MonkeyPatch 环境变量跨测试污染

**现象**：`test_env_file_parsing` 通过 `.env` 文件将 `AGENT_MAX_ITERATIONS=3` 写入 `os.environ`，后续 `test_yaml_base_when_no_env` 继承了该值，导致断言 `max_iterations == 8` 实际为 `3`。

**根因**：`Config._load_env_file()` 直接修改 `os.environ`，测试结束未清理。

**解决方案**：测试 tear-down 中显式清理。

```python
try:
    cfg = Config.from_env(env_file=tmp_path)
    assert cfg.max_iterations == 3
finally:
    os.unlink(tmp_path)
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "AGENT_MAX_ITERATIONS"):
        os.environ.pop(key, None)
```

---

### 问题 T.2 — config.yaml 中的 API key 导致 `test_missing_api_key_raises` 不抛异常

**现象**：config.yaml 里有真实 API key，`test_missing_api_key_raises` 调用 `Config.from_env()` 时 YAML 提供了 key，不抛 `ConfigError`。

**解决方案**：测试中显式传 `yaml_file=None` 跳过 YAML 加载。

```python
Config.from_env(env_file=None, yaml_file=None)
```

---

### 问题 T.3 — 断言逻辑误判：检查消息文本而非存储

**现象**：`assert "买菜" not in window2_text` 失败，因为窗口 2 的 LLM 回复中包含 "没有'买菜'这个任务"。

**根因**：用自然语言文本做精确匹配不可靠。

**解决方案**：直接检查底层 todo 存储。

```python
from src.agent.tools.builtin.todo import _storage as todo_store
tasks1 = {t["task"] for t in todo_store.get("window-1", [])}
tasks2 = {t["task"] for t in todo_store.get("window-2", [])}
assert "买菜" not in tasks2
```

---

## API 相关

### 问题 A.1 — search 工具返回空结果导致 LLM 死循环

**现象**：`LoopError: Agent loop exceeded max iterations (10). Last finish_reason: tool_calls.`

**根因**：DuckDuckGo API 对某些查询（如天气）返回空结果。LLM 反复调用 search 试图获取数据，直到耗尽 `max_iterations`。

**解决方案**：
1. 增大 `max_iterations`（治标）
2. System prompt 增加"工具失败 2 次以上停止重试"（治本）

```
重要：如果工具连续失败 2 次以上，直接告诉用户目前无法获取信息，不要无限重试。
```

---

### 问题 A.2 — API Key 泄露到 Git 历史

**现象**：`config/config.yaml` 中的真实 API key 被提交到 git。

**解决方案**：

```bash
# 1. 创建不含 key 的模板
cp config/config.yaml config/config.example.yaml

# 2. .gitignore 加入
echo "config/config.yaml" >> .gitignore

# 3. 修改已提交的历史
git rm --cached config/config.yaml
git commit --amend
```

**预防**：`.gitignore` 同时忽略 `.env` 和 `config/config.yaml`，仓库中只保留 `.example` 模板。

---

## 汇总统计

| 类别 | 问题数 | 类型 |
|------|--------|------|
| SDK 兼容 | 2 | OpenAI v2.x 异常构造 |
| 接口设计 | 2 | ToolExecutor 兼容、循环导入 |
| Token 计数 | 2 | tiktoken 降级、预估 vs 精确 |
| 数据一致 | 2 | Session 恢复去重、Tool handler 覆盖 |
| 安全 | 1 | Calculator AST 白名单 |
| 异步包装 | 1 | urllib.request → asyncio.to_thread |
| 线程安全 | 1 | Todo 存储 Lock |
| 配置加载 | 3 | YAML 优先级、集成测试 fixture、环境变量污染 |
| 测试 | 3 | monkeypatch 污染、YAML key 干扰、断言误判 |
| API | 2 | search 空结果死循环、key 泄露 |
| Python 兼容 | 1 | asyncio.iscoroutinefunction 废弃 |
| Bug | 1 | except 忘记 as e |

**总计 21 个问题，全部已解决。**
