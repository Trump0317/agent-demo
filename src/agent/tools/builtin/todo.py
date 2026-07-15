"""MVT 3.2 — Todo 内置工具

内存存储的待办事项管理，按 session 隔离。
支持 action: "add" / "list" / "done" / "clear"。
"""

from __future__ import annotations

import threading

# 按 session_id 隔离的存储
_storage: dict[str, list[dict]] = {}
_lock = threading.Lock()


def _get_todos(session_id: str) -> list[dict]:
    """获取 session 的 todos 列表"""
    with _lock:
        if session_id not in _storage:
            _storage[session_id] = []
        return _storage[session_id]


async def todo(
    action: str,
    task: str = "",
    task_id: int = 0,
    session_id: str = "default",
) -> str:
    """管理待办事项

    Args:
        action: 操作类型 — 'add' / 'list' / 'done' / 'clear'
        task: 任务描述（action='add' 时必填）
        task_id: 任务编号（action='done' 时使用）
        session_id: 会话 ID（用于隔离不同会话的 todo）

    Returns:
        操作结果描述
    """
    action = action.strip().lower()
    todos = _get_todos(session_id)

    if action == "add":
        if not task.strip():
            return "Error: task description is required for 'add' action"
        new_id = len(todos) + 1
        todos.append({"id": new_id, "task": task.strip(), "done": False})
        return f"Added todo #{new_id}: {task.strip()}"

    elif action == "list":
        if not todos:
            return "No todos yet."
        lines = []
        for t in todos:
            status = "[x]" if t["done"] else "[ ]"
            lines.append(f"#{t['id']} {status} {t['task']}")
        return "\n".join(lines)

    elif action == "done":
        if task_id <= 0:
            return "Error: valid task_id is required for 'done' action"
        for t in todos:
            if t["id"] == task_id:
                t["done"] = True
                return f"Marked todo #{task_id} as done: {t['task']}"
        return f"Error: todo #{task_id} not found"

    elif action == "clear":
        done_before = sum(1 for t in todos if t["done"])
        _storage[session_id] = [t for t in todos if not t["done"]]
        return f"Cleared {done_before} completed todo(s)."

    else:
        return f"Error: unknown action '{action}'. Supported: add, list, done, clear"
