"""
LangChain / LangGraph 工具的「单一来源」。

tools.py 里 calculator / web_search 是框架无关的纯逻辑（"身体"），
本文件用 @tool 把它们包成 LangChain 工具对象，作为 LangChain 版与 LangGraph 版
共用的唯一来源 —— 不再在两处各写一遍 @tool 定义，避免改了一边忘改另一边。

- 函数名  → 工具名
- docstring  → 工具描述（喂给模型看）
- 类型注解  → 参数 schema（自动生成 JSON Schema）

注意：本文件依赖 langchain_core，所以放在独立模块里，
保持 tools.py 框架无关（自研版 agent.py 只用 openai + ddgs 即可运行）。
"""

from langchain_core.tools import tool

# 工具的"身体"逻辑统一来自 tools.py
from tools import calculator, web_search


@tool
def calculator_lc(expression: str) -> str:
    """安全地计算数学表达式。支持加减乘除、幂、绝对值、取整等。传入一个数学表达式字符串，例如 '3 * (7 + 2) / 4'。"""
    return calculator(expression)


@tool
def web_search_lc(query: str) -> str:
    """在互联网上搜索最新信息。当需要查找实时数据、新闻或事实性知识时使用。"""
    return web_search(query)


# LangChain / LangGraph 入口统一使用的工具列表
LANGCHAIN_TOOLS = [calculator_lc, web_search_lc]
