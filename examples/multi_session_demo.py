"""Multi-session demo: 用户 A 同时开两个独立窗口，互不影响 + Trace 追踪。

窗口 1: 查天气 → 记待办 → 切换回来继续追问
窗口 2: 写周报 → 记待办 → 切换回来继续追问
"""
import asyncio
import os
import sys

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
    config.max_iterations = 15  # 给工具调用留足轮次

    # 用临时目录存 session，演示结束自动清理
    import tempfile
    sessions_dir = tempfile.mkdtemp(prefix="agent-sessions-")
    traces_dir = tempfile.mkdtemp(prefix="agent-traces-")

    # 覆写路径配置
    config.sessions_dir = sessions_dir
    config.traces_dir = traces_dir

    store = JsonSessionStore(base_dir=sessions_dir, config=config)

    # ================================================================
    # 窗口 1 — 查天气 + 记待办
    # ================================================================
    print_divider("窗口 1 创建 — session_id=window-1")
    agent1 = Agent(
        config=config,
        system_prompt=(
            "你是一个生活助理。有 search 和 todo 工具可用。"
            "重要：如果工具连续失败 2 次以上，直接告诉用户目前无法获取信息，不要无限重试。"
        ),
        session_id="window-1",
        session_store=store,
        trace_enabled=True,
    )

    print("📝 窗口 1 第一轮: 查北京天气，如果下雨就 add 到待办")
    r1 = await agent1.run("请查询一下北京的天气，如果可能下雨就把'带伞'加到我的待办里。")
    print(f"🤖 {r1.content}")
    if r1.usage:
        print(f"   [tokens: {r1.usage.total_tokens}]")

    print("\n📝 窗口 1 第二轮: 追加待办 + 追问天气")
    r2 = await agent1.run("再帮我加一条'买牛奶'到待办。然后告诉我现在的待办有哪些？")
    print(f"🤖 {r2.content}")
    if r2.usage:
        print(f"   [tokens: {r2.usage.total_tokens}]")

    # ================================================================
    # 窗口 2 — 写周报 + 记待办
    # ================================================================
    print_divider("窗口 2 创建 — session_id=window-2（独立会话）")
    agent2 = Agent(
        config=config,
        system_prompt=(
            "你是一个工作助理。有 search 和 todo 工具可用。"
            "重要：如果工具连续失败 2 次以上，直接告诉用户目前无法获取信息，不要无限重试。"
        ),
        session_id="window-2",
        session_store=store,
        trace_enabled=True,
    )

    print("📝 窗口 2 第一轮: 写周报 + 记待办")
    r3 = await agent2.run(
        "帮我写一段本周工作总结的草稿（内容：完成了项目A的测试、修复了3个线上bug）。"
        "然后把'提交周报'加到我的待办里。"
    )
    print(f"🤖 {r3.content}")
    if r3.usage:
        print(f"   [tokens: {r3.usage.total_tokens}]")

    print("\n📝 窗口 2 第二轮: 追加待办")
    r4 = await agent2.run("再加一条'预约下周一的会议'。看看我的待办有哪些？")
    print(f"🤖 {r4.content}")
    if r4.usage:
        print(f"   [tokens: {r4.usage.total_tokens}]")

    # ================================================================
    # 切回窗口 1 — 验证记忆独立
    # ================================================================
    print_divider("切回窗口 1 — 验证是否记得之前的上下文")
    print("📝 窗口 1 第三轮: 追问之前的内容")
    r5 = await agent1.run("之前我说的天气怎么样了？我是否已经加了'带伞'？")
    print(f"🤖 {r5.content}")
    if r5.usage:
        print(f"   [tokens: {r5.usage.total_tokens}]")

    # ================================================================
    # 切回窗口 2 — 验证记忆独立
    # ================================================================
    print_divider("切回窗口 2 — 验证是否记得之前的上下文")
    print("📝 窗口 2 第三轮: 追问之前的内容")
    r6 = await agent2.run("我之前让你写的周报草稿是什么内容？我的待办有哪些？")
    print(f"🤖 {r6.content}")
    if r6.usage:
        print(f"   [tokens: {r6.usage.total_tokens}]")

    # ================================================================
    # 验证隔离
    # ================================================================
    print_divider("Session 隔离验证")

    msgs1 = agent1.messages
    msgs2 = agent2.messages

    window1_text = " ".join(m.content for m in msgs1 if m.content)
    window2_text = " ".join(m.content for m in msgs2 if m.content)

    print(f"窗口 1 消息数: {len(msgs1)}, token: {agent1.token_count}")
    print(f"窗口 2 消息数: {len(msgs2)}, token: {agent2.token_count}")

    # 窗口 1 的话题（天气、带伞、买牛奶）不应出现在窗口 2
    assert "天气" in window1_text or "weather" in window1_text.lower()
    assert "带伞" in window1_text or "买牛奶" in window1_text
    assert "天气" not in window2_text and "带伞" not in window2_text, \
        "FAIL: 窗口 2 泄露了窗口 1 的内容!"

    # 窗口 2 的话题（周报、项目A、预约会议）不应出现在窗口 1
    assert "周报" in window2_text or "项目" in window2_text
    assert "周报" not in window1_text and "项目A" not in window1_text, \
        "FAIL: 窗口 1 泄露了窗口 2 的内容!"

    print("✅ Session 隔离验证通过：两个窗口互不干扰")

    # ================================================================
    # Trace 输出
    # ================================================================
    print_divider("Trace 文件")
    import glob
    trace_files = sorted(glob.glob(os.path.join(traces_dir, "*.jsonl")))
    for tf in trace_files:
        print(f"\n📄 {os.path.basename(tf)}")
        with open(tf) as f:
            lines = f.readlines()
        print(f"   共 {len(lines)} 条 Trace 记录")
        for i, line in enumerate(lines[:3]):  # 只展示前 3 条
            import json
            record = json.loads(line)
            print(f"   [{record['phase']}] {list(record['data'].keys())}")
        if len(lines) > 3:
            print(f"   ... 还有 {len(lines) - 3} 条 ...")

    # 清理临时目录
    import shutil
    shutil.rmtree(sessions_dir, ignore_errors=True)
    shutil.rmtree(traces_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
