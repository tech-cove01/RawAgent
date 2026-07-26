"""
Agent 工具模块
提供 calculator（安全算数）和 web_search（联网搜索）两个工具
"""

import ast
import operator
import json
from ddgs import DDGS

# ============================================================
# 工具函数
# ============================================================

def calculator(expression: str) -> str:
    """
    安全地计算数学表达式。
    使用 ast 解析而非 eval，避免代码注入风险。
    """
    # 只允许数学相关节点
    allowed_nodes = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.USub, ast.UAdd,
        ast.Num,  # Python < 3.8
        ast.Constant,  # Python 3.8+
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.Call,
    }

    # 允许的安全函数
    safe_funcs = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "int": int, "float": float,
    }

    try:
        tree = ast.parse(expression.strip(), mode="eval")

        # 检查所有节点是否安全
        for node in ast.walk(tree):
            if type(node) not in allowed_nodes:
                return f"错误：不允许的操作 `{type(node).__name__}`"
            if isinstance(node, ast.Call):
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name and func_name not in safe_funcs:
                    return f"错误：不允许的函数 `{func_name}`"

        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, safe_funcs)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"


def web_search(query: str) -> str:
    """
    使用 DuckDuckGo 进行联网搜索，返回前 5 条结果的摘要。
    免费免 key，无需额外注册。
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "未找到相关搜索结果。"

        output = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "无标题")
            href = r.get("href", "")
            body = r.get("body", "")[:200]  # 截断过长内容
            output.append(f"{i}. {title}\n   链接：{href}\n   摘要：{body}")
        return "\n\n".join(output)
    except Exception as e:
        return f"搜索失败：{e}"


# ============================================================
# OpenAI Function Calling 格式的工具 Schema
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算。支持加减乘除、幂、绝对值、取整等。传入一个数学表达式字符串，例如 '3 * (7 + 2) / 4'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，例如 '3*7+2' 或 'sqrt(4) + abs(-5)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "在互联网上搜索最新信息。当需要查找实时数据、新闻、或不知道的事实性知识时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，用简洁的语言描述你要查找的内容",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ============================================================
# 工具调度
# ============================================================

def execute_tool(name: str, arguments: str) -> str:
    """根据工具名和 JSON 参数执行对应工具，返回字符串结果。"""
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return f"参数解析失败：{arguments}"

    if name == "calculator":
        return calculator(args.get("expression", ""))
    elif name == "web_search":
        return web_search(args.get("query", ""))
    else:
        return f"未知工具：{name}"
