"""MVT 3.2 — Search 内置工具

使用 urllib.request + DuckDuckGo Instant Answer API 搜索。
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

DUCKDUCKGO_API = "https://api.duckduckgo.com/"


async def search(query: str) -> str:
    """搜索 DuckDuckGo Instant Answer API

    Args:
        query: 搜索查询字符串

    Returns:
        搜索结果摘要
    """
    import asyncio

    query = query.strip()
    if not query:
        return "Error: empty search query"

    params = urllib.parse.urlencode({
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
    })
    url = f"{DUCKDUCKGO_API}?{params}"

    try:
        # 使用 asyncio.to_thread 异步执行 HTTP 请求
        response_text = await asyncio.to_thread(_fetch_url, url, timeout=15)
    except Exception as e:
        logger.warning(f"Search request failed: {e}")
        return f"Error: search request failed — {e}"

    try:
        data = json.loads(response_text)
    except json.JSONDecodeError:
        return "Error: failed to parse search response"

    # 提取结果
    parts: list[str] = []

    # AbstractText（主要摘要）
    abstract = data.get("AbstractText", "")
    if abstract:
        parts.append(abstract)

    # RelatedTopics（相关主题）
    related = data.get("RelatedTopics", [])
    for topic in related[:5]:  # 最多 5 条
        if isinstance(topic, dict) and topic.get("Text"):
            parts.append(f"- {topic['Text']}")

    if not parts:
        heading = data.get("Heading", "")
        if heading:
            return f"No detailed results for '{heading}'. Try a different query."
        return f"No results found for '{query}'."

    return "\n".join(parts)


def _fetch_url(url: str, timeout: float) -> str:
    """同步 fetch URL（在 asyncio.to_thread 中运行）"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "AgentDemo/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")
