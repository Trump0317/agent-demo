"""MVT 3.2 — Calculator 内置工具

使用 ast.parse + ast.NodeVisitor 安全求值算术表达式。
支持：+, -, *, /, **, //, %, (), 整数和浮点数。
禁止：函数调用、属性访问、导入语句等。
"""

from __future__ import annotations

import ast
import math
import operator

# 安全二元运算符白名单
_SAFE_OPS: dict[type[ast.operator], callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 安全常量类型
_SAFE_CONSTANTS = (int, float)


class _CalcVisitor(ast.NodeVisitor):
    """安全遍历 AST，仅允许安全的算术运算"""

    def __init__(self) -> None:
        self.result: float | None = None

    def visit_Expression(self, node: ast.Expression) -> None:
        self.result = self._eval(node.body)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = self._eval(node.operand)
        return op_func(operand)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        op_func = _SAFE_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = self._eval(node.left)
        right = self._eval(node.right)
        return op_func(left, right)

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, _SAFE_CONSTANTS):
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        return float(node.value)

    def generic_visit(self, node: ast.AST) -> None:
        """拒绝所有非允许的 AST 节点"""
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def _eval(self, node: ast.AST) -> float:
        """递归求值 AST 节点"""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        result = visitor(node)
        if result is None:
            raise ValueError(f"Unexpected None from {method}")
        return result


async def calculator(expression: str) -> str:
    """安全计算算术表达式

    支持：+, -, *, /, **, //, %, (), 整数和浮点数。
    拒绝：函数调用、属性访问、导入语句。

    Args:
        expression: 算术表达式字符串，如 "2 + 3 * 4"

    Returns:
        计算结果字符串
    """
    # 预处理：替换常见 Unicode 字符
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Error: invalid syntax — {e}"

    visitor = _CalcVisitor()
    try:
        visitor.visit(tree)
        if visitor.result is None:
            return "Error: evaluation produced no result"
        # 美化为整数（如果结果恰为整数）
        if visitor.result == int(visitor.result):
            return str(int(visitor.result))
        return str(visitor.result)
    except Exception as e:
        return f"Error: {e}"
