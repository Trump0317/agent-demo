"""Tests for MVT 3.2 — Built-in tools (calculator, search, todo)."""
import pytest

from src.agent.tools.builtin.calculator import calculator
from src.agent.tools.builtin.search import search
from src.agent.tools.builtin.todo import todo, _storage as todo_storage


class TestCalculator:
    """Calculator 工具测试"""

    @pytest.mark.asyncio
    async def test_simple_addition(self):
        result = await calculator("1 + 1")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        result = await calculator("2 + 3 * 4")
        assert result == "14"

    @pytest.mark.asyncio
    async def test_parentheses(self):
        result = await calculator("(2 + 3) * 4")
        assert result == "20"

    @pytest.mark.asyncio
    async def test_float_result(self):
        result = await calculator("5 / 2")
        assert result == "2.5"

    @pytest.mark.asyncio
    async def test_power(self):
        result = await calculator("2 ** 10")
        assert result == "1024"

    @pytest.mark.asyncio
    async def test_negative_number(self):
        result = await calculator("-3 + 5")
        assert result == "2"

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        result = await calculator("1 / 0")
        assert "Error" in result or "division" in result.lower()

    @pytest.mark.asyncio
    async def test_blocks_import(self):
        """禁止 __import__"""
        result = await calculator("__import__('os').system('ls')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_blocks_eval(self):
        """禁止 eval（这是一个数字表达式，但 'eval' 作为变量不合法）"""
        result = await calculator("eval('1+1')")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_blocks_attribute_access(self):
        """禁止属性访问"""
        result = await calculator("[].__class__.__mro__[1].__subclasses__()")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        result = await calculator("")
        assert "Error" in result


class TestSearch:
    """Search 工具测试"""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        result = await search("")
        assert "Error" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_search(self):
        """真实搜索测试（需要网络连接）"""
        result = await search("Python programming language")
        assert len(result) > 0
        assert "Error" not in result or "No results found" in result


class TestTodo:
    """Todo 工具测试"""

    def setup_method(self):
        """在每个测试前清空 todo 存储"""
        todo_storage.clear()

    @pytest.mark.asyncio
    async def test_add_and_list(self):
        result = await todo("add", task="Buy milk", session_id="test")
        assert "Added" in result

        result = await todo("list", session_id="test")
        assert "Buy milk" in result
        assert "[ ]" in result

    @pytest.mark.asyncio
    async def test_add_without_task(self):
        result = await todo("add", task="", session_id="test")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_done(self):
        await todo("add", task="Task 1", session_id="test")
        await todo("add", task="Task 2", session_id="test")

        result = await todo("done", task_id=1, session_id="test")
        assert "done" in result.lower()

        list_result = await todo("list", session_id="test")
        assert "[x]" in list_result  # 已完成标记
        assert "[ ]" in list_result  # 未完成

    @pytest.mark.asyncio
    async def test_done_nonexistent(self):
        result = await todo("done", task_id=999, session_id="test")
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_clear(self):
        await todo("add", task="Task 1", session_id="test")
        await todo("add", task="Task 2", session_id="test")
        await todo("done", task_id=1, session_id="test")

        result = await todo("clear", session_id="test")
        assert "Cleared 1" in result

        list_result = await todo("list", session_id="test")
        assert "Task 2" in list_result
        assert "Task 1" not in list_result

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        result = await todo("unknown", session_id="test")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """不同 session 的 todo 隔离"""
        await todo("add", task="Task A", session_id="s1")
        await todo("add", task="Task B", session_id="s2")

        list1 = await todo("list", session_id="s1")
        list2 = await todo("list", session_id="s2")

        assert "Task A" in list1
        assert "Task B" not in list1
        assert "Task B" in list2
        assert "Task A" not in list2
