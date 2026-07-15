# AI Prompt 与问题解决记录

## 项目概述

项目 Prompt 定义在 `prompt.md`，完整描述了三阶段（Phase 1/2/3）的目标和验收标准。以下记录开发过程中遇到的典型问题和解决方式。

---

## Phase 1: Foundation + Agent Loop

### 问题 1：OpenAI SDK 异常初始化需要 httpx.Response

**现象**：Mock `AuthenticationError("401")` 直接抛异常时报 `TypeError: APIStatusError.__init__() missing 2 required keyword-only arguments: 'response' and 'body'`

**原因**：OpenAI v2.x SDK 的异常类需要合法的 `httpx.Response` 对象才能实例化。

**解决**：构造辅助函数 `_make_mock_response()` 生成合法的 mock `httpx.Response`，再传入异常构造函数。

```python
def _make_mock_response(status_code: int = 200) -> httpx.Response:
    request = httpx.Request("POST", "https://api.deepseek.com/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)

AuthenticationError("msg", response=_make_mock_response(401), body=None)
```

---

### 问题 2：ToolExecutor 与 ToolRegistry 的接口兼容

**设计决策**：Phase 1 定义了 `ToolExecutor` ABC（带 Stub），Phase 3 的 `ToolRegistry` 需要继承它，保证 `AgentLoop` 无需修改。

**解决**：`ToolRegistry` 直接继承 `ToolExecutor` ABC，实现 `execute()` 和 `list_schemas()`。Phase 3 只需将 `AgentLoop(tool_executor=registry)` 传入即可，实现了真正的"接口替换不破坏已有代码"。

---

### 问题 3：TraceCollector 与 AgentLoop 的循环导入

**现象**：`loop.py` 的 `AgentLoop.__init__` 需要 `TraceCollector` 类型提示，而 `trace.py` 未来可能需要引用 `loop.py`。

**解决**：使用 `TYPE_CHECKING` 和字符串类型提示延迟导入：

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.trace import TraceCollector

class AgentLoop:
    def __init__(self, ..., trace_collector: "TraceCollector | None" = None):
        ...
```

---

## Phase 2: Session Management

### 问题 4：tiktoken 安装与降级

**现象**：无 `tiktoken` 时 `count_tokens` 抛 `ImportError`。

**解决**：采用 try/except 导入，不可用时降级为字符估算（`len(text) // 4`），并记录 warning 日志。

---

### 问题 5：Session 恢复时的 system prompt 去重

**现象**：`AgentSession.__init__` 根据 `system_prompt` 自动插入 system 消息，`from_dict()` 恢复时再次调用导致重复。

**解决**：`from_dict()` 从 dict 恢复消息时，跳过首条 system 消息，依赖 `__init__` 自动插入。

---

### 问题 6：AgentLoop 与 AgentSession 的 token 计数同步

**设计决策**：LLM 调用后的 `response.usage` 是精确值，但 Session 内的 `count_tokens()` 是预估值。如何权衡？

**解决**：双路径：
1. `add_message()` 后自动累积预估 token 数（用于压缩判断，速度快）
2. LLM 调用后，`session.set_token_count(response.usage.total_tokens)` 覆盖为精确值（用于状态展示）

---

## Phase 3: Tool System

### 问题 7：Calculator 的 AST 安全求值

**现象**：如何安全求值用户输入的算术表达式，阻止 `__import__('os').system('rm -rf /')`？

**解决**：使用 `ast.parse(mode="eval")` + 自定义 `NodeVisitor`，仅允许：
- 二元运算：`Add, Sub, Mult, Div, Pow, FloorDiv, Mod`
- 一元运算：`USub, UAdd`
- 常量：`int, float`

所有其他 AST 节点（`Call`, `Attribute`, `Import` 等）一律通过 `generic_visit()` 拒绝。

---

### 问题 8：search 工具的异步 HTTP 请求

**设计决策**：Prompt 要求使用标准库 `urllib.request`（同步），但 Agent 全链路异步。

**解决**：使用 `asyncio.to_thread()` 包装同步 HTTP 请求：

```python
response_text = await asyncio.to_thread(_fetch_url, url, timeout=15)
```

---

### 问题 9：Todo 的多 Session 隔离 + 线程安全

**现象**：`todo` 工具是内存存储，多个 Agent 实例并发调用时需保证隔离与安全。

**解决**：
1. 全局 `dict[str, list[dict]]` 按 `session_id` 分组
2. `threading.Lock` 保护写操作
3. `todo()` handler 接收 `session_id` 参数，LLM 调用时自动传入

---

## Phase 4: Agent 组装层

### 问题 10：add_tool 后工具未生效

**现象**：先 `agent.add_tool(new_tool)` 后 `agent.run("task")`，LLM 调用时 tools 参数不包含新工具。

**原因**：`add_tool` 更新了 `ToolRegistry`，但 `AgentLoop` 在 `_init()` 时已缓存了 `tools_schemas`。

**解决**：`run()` 调用时从 `ToolRegistry.list_schemas()` 实时获取（而非缓存），确保动态注册即时生效。

---

### 问题 11：YAML 配置加载与 .env 优先级

**现象**：`from_yaml()` 的字段被 `.env` 或环境变量覆盖，优先级行为不一致。

**解决**：实现三级加载链：

```
环境变量 > .env 文件 > config/config.yaml > 默认值
```

`from_env()` 优先级：
1. 用 `from_yaml()` 获取 YAML 基础值
2. `.env` 文件写入 `os.environ`（不覆盖已有环境变量，即 YAML < .env 已达）
3. 环境变量读取时以 YAML 值作为 fallback → 最终 `env > .env > yaml`

---

## 测试相关

### 问题 12：MonkeyPatch 环境变量的跨测试污染

**现象**：`test_env_file_parsing` 加载 `.env` 文件后，`AGENT_MAX_ITERATIONS=3` 残留于 `os.environ`，导致后续 `test_yaml_base_when_no_env` 失败。

**解决**：在测试 tear-down 中显式 `os.environ.pop(key, None)` 清理，或使用 `monkeypatch` 隔离。

---

### 问题 13：Integration 测试与 CI 环境

**设计决策**：集成测试依赖 `DEEPSEEK_API_KEY` + 网络，不应在 CI 中自动运行。

**解决**：使用 `@pytest.mark.integration` 标记，CI 运行 `pytest -m "not integration"`。

---

## 总结

| 类别 | 问题数 | 典型解决方式 |
|------|--------|-------------|
| 异常处理 | 2 | 构造合法 mock 对象 |
| 接口兼容 | 1 | ABC 继承 + 替换不破坏 |
| 循环导入 | 1 | TYPE_CHECKING + 字符串类型提示 |
| Token 计数 | 1 | try/except 降级 |
| 序列化 | 1 | 去重逻辑 + from_dict 跳过 |
| 安全 | 1 | AST NodeVisitor 白名单 |
| 异步包装 | 1 | asyncio.to_thread |
| 线程安全 | 1 | Lock + 按 key 分组 |
| 配置加载 | 2 | 三级优先级链 |
| 测试隔离 | 2 | monkeypatch + 手动清理 |
| CI 兼容 | 1 | pytest markers |
