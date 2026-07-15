"""Multi-session demo: 用户 A 同时开两个独立窗口，互不影响 + Trace 追踪。

窗口 1: 记待办 → 追问
窗口 2: 写周报记待办 → 追问

Trace 文件保存在 ./traces/ 目录，可以直接查看。
"""
import asyncio
import os
import sys
import json
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import Agent
from src.agent.persistence import JsonSessionStore
from src.agent.config import Config


def print_divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


async def main() -> None:
    config = Config.from_env()
    config.max_iterations = 10

    store = JsonSessionStore(base_dir=config.sessions_dir, config=config)

    # ================================================================
    # 窗口 1 — 记待办
    # ================================================================
    print_divider("窗口 1 创建 — session_id=window-1")
    agent1 = Agent(
        config=config,
        system_prompt="你是生活助理。用 todo 工具管理待办事项。",
        session_id="window-1",
        session_store=store,
        trace_enabled=True,
    )

    print("📝 窗口 1: 添加生活待办")
    r1 = await agent1.run("帮我把'买菜'、'取快递'、'交电费'加到待办里，然后告诉我有哪些。")
    print(f"🤖 {r1.content}")

    print("\n📝 窗口 1: 追问")
    r2 = await agent1.run("把'取快递'标记为完成，然后告诉我还有哪些没做。")
    print(f"🤖 {r2.content}")

    # ================================================================
    # 窗口 2 — 写周报记待办
    # ================================================================
    print_divider("窗口 2 创建 — session_id=window-2（独立会话）")
    agent2 = Agent(
        config=config,
        system_prompt="你是工作助理。用 todo 工具管理待办事项。",
        session_id="window-2",
        session_store=store,
        trace_enabled=True,
    )

    print("📝 窗口 2: 添加工作待办")
    r3 = await agent2.run("帮我把'提交周报'、'预约下周会议'、'代码review'加到待办，然后告诉我有哪些。")
    print(f"🤖 {r3.content}")

    print("\n📝 窗口 2: 追问")
    r4 = await agent2.run("把'代码review'标记完成，再告诉我当前待办列表。")
    print(f"🤖 {r4.content}")

    # ================================================================
    # 切回窗口 1 — 验证记忆独立
    # ================================================================
    print_divider("切回窗口 1 — 验证是否记得之前的上下文，没有混入窗口2的内容")
    print("📝 窗口 1: 追问")
    r5 = await agent1.run("我刚才的待办有哪些？有没有'提交周报'这个任务？")
    print(f"🤖 {r5.content}")

    # ================================================================
    # 切回窗口 2 — 验证记忆独立
    # ================================================================
    print_divider("切回窗口 2 — 验证是否记得之前的上下文，没有混入窗口1的内容")
    print("📝 窗口 2: 追问")
    r6 = await agent2.run("我刚才的待办有哪些？有没有'买菜'这个任务？")
    print(f"🤖 {r6.content}")

    # ================================================================
    # 验证隔离
    # ================================================================
    print_divider("Session 隔离验证")

    msgs1 = agent1.messages
    msgs2 = agent2.messages

    print(f"窗口 1 消息数: {len(msgs1)}, token: {agent1.token_count}")
    print(f"窗口 2 消息数: {len(msgs2)}, token: {agent2.token_count}")

    # 直接检查 todo 存储，验证窗口隔离
    from src.agent.tools.builtin.todo import _storage as todo_store

    todos1 = todo_store.get("window-1", [])
    todos2 = todo_store.get("window-2", [])

    tasks1 = {t["task"] for t in todos1}
    tasks2 = {t["task"] for t in todos2}

    print(f"\n窗口 1 待办存储: {tasks1}")
    print(f"窗口 2 待办存储: {tasks2}")

    # 窗口 1 的生活待办不应出现在窗口 2
    assert "买菜" in tasks1 and "取快递" in tasks1 and "交电费" in tasks1, \
        f"窗口 1 待办不完整: {tasks1}"
    assert "买菜" not in tasks2 and "取快递" not in tasks2 and "交电费" not in tasks2, \
        f"FAIL: 窗口 2 泄露了窗口 1 的待办! {tasks2}"

    # 窗口 2 的工作待办不应出现在窗口 1
    assert "提交周报" in tasks2 and "预约下周会议" in tasks2 and "代码review" in tasks2, \
        f"窗口 2 待办不完整: {tasks2}"
    assert "提交周报" not in tasks1 and "预约下周会议" not in tasks1 and "代码review" not in tasks1, \
        f"FAIL: 窗口 1 泄露了窗口 2 的待办! {tasks1}"

    print("\n✅ Session 隔离验证通过：两个窗口互不干扰")

    # ================================================================
    # Trace 输出
    # ================================================================
    print_divider("Trace 文件（保存在 ./traces/ 目录）")

    trace_files = sorted(glob.glob(os.path.join(config.traces_dir, "*.jsonl")))
    for tf in trace_files:
        with open(tf) as f:
            lines = f.readlines()
        print(f"\n📄 {os.path.basename(tf)}  ({len(lines)} 条记录)")
        print(f"   路径: {tf}")

        # 展示每个 phase 的摘要
        phases = {}
        for line in lines:
            r = json.loads(line)
            phase = r["phase"]
            if phase not in phases:
                phases[phase] = 0
            phases[phase] += 1

        print("   Phase 分布:")
        for phase, count in phases.items():
            print(f"     {phase}: {count}")

        # 展示前 3 条详细内容
        print("   前 3 条详情:")
        for line in lines[:3]:
            r = json.loads(line)
            print(f"     [{r['phase']}] {json.dumps(r['data'], ensure_ascii=False)[:120]}...")


if __name__ == "__main__":
    asyncio.run(main())
