"""Basic usage example — demonstrates the Phase 4 unified Agent API."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import Agent


async def main() -> None:
    # ==================== 最简用法 ====================
    # 一行创建（自动加载 .env 配置 + 预装全部 3 个内置工具）
    agent = Agent()

    tasks = [
        "Calculate 123 * 456 and tell me the result.",
        "Add 'buy groceries' and 'call dentist' to my todo list, then show me all todos.",
    ]

    print("=" * 60)
    print("Agent Demo")
    print("=" * 60)

    for task in tasks:
        print(f"\n📝 Task: {task}")
        print("-" * 40)
        result = await agent.run(task)
        if result.content:
            print(f"🤖 Response: {result.content}")
        if result.usage:
            print(f"   Tokens: {result.usage.total_tokens} "
                  f"(prompt={result.usage.prompt_tokens}, "
                  f"completion={result.usage.completion_tokens})")

    print("\n" + "=" * 60)
    print(f"Session messages: {len(agent.messages)}")
    print(f"Session tokens:  {agent.token_count}")


async def demo_advanced() -> None:
    """演示精细控制的用法"""
    from src.agent.agent import BUILTIN_CALCULATOR, BUILTIN_SEARCH, BUILTIN_TODO, create_agent
    from src.agent.persistence import JsonSessionStore
    from src.agent.config import Config

    config = Config.from_env()

    # 方式一：create_agent 工厂函数
    agent = create_agent(
        system_prompt="You are a math tutor. Use the calculator tool for math problems.",
        tools=[BUILTIN_CALCULATOR],  # 只装 calculator
        compression="truncate",
        persist=True,  # 自动保存 session
    )

    result = await agent.run("What is the square root of 144?")
    print(f"\nMath tutor: {result.content}")

    await agent.reset()

    # 方式二：Agent 数据类直接构造
    store = JsonSessionStore(base_dir=config.sessions_dir, config=config)

    agent2 = Agent(
        system_prompt="You are a research assistant.",
        tools=[BUILTIN_SEARCH, BUILTIN_CALCULATOR],
        compression="summarize",
        session_store=store,
        session_id="research-session",  # 可恢复历史
    )

    result2 = await agent2.run("Search for 'latest Python 3.14 features' and summarize.")
    print(f"\nResearch assistant: {result2.content}")


if __name__ == "__main__":
    asyncio.run(main())
