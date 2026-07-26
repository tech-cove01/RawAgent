"""
后端（引擎）插件包。

四个框架（raw / LangChain / LangGraph / OpenAI-Agents）各自实现同一个
AgentBackend 接口，只负责「驱动 agent 跑工具循环」这一层 glue 代码。
系统提示词与对话主循环统一在 main.py，不在这里重复。
"""

from .base import AgentBackend
from .raw import RawBackend
from .langchain import LangChainBackend
from .langgraph import LangGraphBackend
from .openai_agents import OpenAIAgentsBackend

# 名称 → 后端类，工厂按命令行 --backend 查表
_BACKENDS = {
    "raw": RawBackend,
    "langchain": LangChainBackend,
    "langgraph": LangGraphBackend,
    "agents": OpenAIAgentsBackend,
}


def create_backend(name: str, model: str, system_prompt: str) -> AgentBackend:
    """根据名字创建对应的后端实例。"""
    if name not in _BACKENDS:
        raise ValueError(
            f"未知后端: {name!r}，可选: {list(_BACKENDS)}"
        )
    return _BACKENDS[name](model, system_prompt)


__all__ = ["create_backend", "AgentBackend"]
