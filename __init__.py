"""
后端（引擎）插件包。

四个框架（raw / LangChain / LangGraph / OpenAI-Agents）各自实现同一个
AgentBackend 接口，只负责「驱动 agent 跑工具循环」这一层 glue 代码。
系统提示词与对话主循环统一在 main.py，不在这里重复。
"""

from .base import AgentBackend
from .raw import RawBackend

# 注意：langchain / langgraph / openai_agents 三个后端**不**在模块加载时导入，
# 而是惰性导入——只有 create_backend 真正选中它们时才加载对应模块。
# 这样即使当前环境没装对应框架（或版本不兼容），默认的 raw 引擎也能正常启动，
# 同时这三个文件作为「用过这些框架」的实现证据仍保留在仓库中。
_BACKEND_IMPORTS = {
    "raw": ("RawBackend", ".raw"),
    "langchain": ("LangChainBackend", ".langchain"),
    "langgraph": ("LangGraphBackend", ".langgraph"),
    "agents": ("OpenAIAgentsBackend", ".openai_agents"),
}


def create_backend(name: str, model: str, system_prompt: str, retriever=None) -> AgentBackend:
    """根据名字创建对应的后端实例（惰性导入对应模块）。"""
    if name not in _BACKEND_IMPORTS:
        raise ValueError(
            f"未知后端: {name!r}，可选: {list(_BACKEND_IMPORTS)}"
        )
    class_name, module = _BACKEND_IMPORTS[name]
    # raw 已顶层导入，其余按需导入，避免框架缺失时拖累整个包加载
    if name == "raw":
        # raw 唯一接入 RAG 检索器
        return RawBackend(model, system_prompt, retriever=retriever)
    import importlib
    backend_mod = importlib.import_module(module, __package__)
    backend_cls = getattr(backend_mod, class_name)
    return backend_cls(model, system_prompt)


__all__ = ["create_backend", "AgentBackend"]
