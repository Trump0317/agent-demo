"""Interactive Agent CLI — 一个终端一个窗口。

用法:
    python examples/cli.py                  # 自动生成 session_id
    python examples/cli.py --session work   # 指定窗口名 (可恢复历史)
    python examples/cli.py --session life --system "你是生活助理"
"""
import asyncio
import os
import sys
import argparse
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import Agent
from src.agent.persistence import JsonSessionStore
from src.agent.config import Config


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"


def print_banner(session_id: str, system_prompt: str) -> None:
    width = 56
    print(f"{CYAN}{BOLD}")
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║{'Agent CLI'.center(width - 2)}║")
    print("╠" + "═" * (width - 2) + "╣")
    print(f"║  session: {session_id:<{width - 14}}║")
    prompt_short = system_prompt[:width - 14].ljust(width - 14)
    print(f"║  system:  {prompt_short}║")
    print("╠" + "═" * (width - 2) + "╣")
    print(f"║  /exit  退出    /reset  清空历史          ║")
    print(f"║  /trace 查看 trace 文件路径               ║")
    print("╚" + "═" * (width - 2) + "╝")
    print(f"{RESET}")


def print_agent(content: str) -> None:
    """打印 Agent 回复，自动换行"""
    wrapper = textwrap.TextWrapper(width=72, initial_indent="", subsequent_indent="  ")
    for line in content.split("\n"):
        if line.strip():
            wrapped = wrapper.fill(line)
            print(f"{GREEN}{wrapped}{RESET}")
        else:
            print()


def print_info(msg: str) -> None:
    print(f"{DIM}{msg}{RESET}")


def print_error(msg: str) -> None:
    print(f"{YELLOW}{msg}{RESET}")


async def interactive_loop(agent: Agent, session_id: str, traces_dir: str) -> None:
    """交互式主循环"""
    print_info("输入你的问题 (或 /exit 退出):\n")

    while True:
        try:
            user_input = input(f"{CYAN}{BOLD}你>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n")
            break

        if not user_input:
            continue

        # ── 内置命令 ──
        if user_input.startswith("/"):
            cmd = user_input[1:].strip().lower()
            if cmd in ("exit", "quit", "q"):
                print_info("再见!")
                break
            elif cmd == "reset":
                await agent.reset()
                print_info("✅ 会话已清空 (system prompt 保留)")
                continue
            elif cmd == "trace":
                trace_path = os.path.join(traces_dir, f"{session_id}.jsonl")
                if os.path.isfile(trace_path):
                    size = os.path.getsize(trace_path)
                    lines = sum(1 for _ in open(trace_path))
                    print_info(f"📄 {trace_path}")
                    print_info(f"   {lines} 条记录, {size} bytes")
                else:
                    print_info("尚未生成 trace 文件")
                continue
            elif cmd == "help":
                print_info("可用命令: /exit, /reset, /trace, /help")
                continue
            else:
                print_error(f"未知命令: /{cmd}, 输入 /help 查看帮助")
                continue

        # ── 调用 Agent ──
        print_info("⏳ 思考中...")
        try:
            response = await agent.run(user_input)
        except Exception as e:
            print_error(f"❌ 出错: {e}")
            continue

        if response.content:
            print_agent(response.content)
            if response.usage:
                print_info(f"[tokens: {response.usage.total_tokens} | "
                           f"prompt={response.usage.prompt_tokens} "
                           f"completion={response.usage.completion_tokens}]")
        print()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI — 一个终端一个窗口")
    parser.add_argument("--session", "-s", type=str, default=None,
                        help="会话 ID (用于恢复历史)")
    parser.add_argument("--system", "-p", type=str,
                        default="你是一个有用的 AI 助手，可以用 calculator/search/todo 工具。",
                        help="系统提示词")
    parser.add_argument("--no-trace", action="store_true", help="关闭 trace")
    parser.add_argument("--compress", choices=["truncate", "summarize"], default=None,
                        help="上下文压缩策略")
    args = parser.parse_args()

    # 加载配置
    config = Config.from_env()

    # 生成 session_id
    import uuid
    session_id = args.session or uuid.uuid4().hex[:8]

    # 尝试恢复已保存的 session
    store = JsonSessionStore(base_dir=config.sessions_dir, config=config)
    resumed = False
    try:
        existing = await store.load(session_id)
        resumed = len(existing.messages) > 1
    except Exception:
        pass

    # 创建 Agent
    agent = Agent(
        config=config,
        system_prompt=args.system,
        session_id=session_id,
        session_store=store,
        trace_enabled=not args.no_trace,
        compression=args.compress,
    )

    # 显示 banner
    print_banner(session_id, args.system)
    if resumed:
        print_info(f"📂 已恢复历史会话 ({len(agent._session.messages) if agent._session else 0} 条消息)")

    await interactive_loop(agent, session_id, config.traces_dir)


if __name__ == "__main__":
    asyncio.run(main())
