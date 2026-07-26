"""
后端③：LangGraph 状态图引擎（create_react_agent）。
与旧 agent_langgraph.py 等价，系统提示词上交给 main.py。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from tools_langchain import LANGCHAIN_TOOLS

from .base import AgentBackend

load_dotenv()


class LangGraphBackend(AgentBackend):
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            temperature=0,
        )
        checkpointer = MemorySaver()
        self.app = create_react_agent(
            llm,
            LANGCHAIN_TOOLS,
            prompt=system_prompt,        # 系统提示（新版支持直接传字符串）
            checkpointer=checkpointer,
        )
        # thread_id 相当于会话 ID，MemorySaver 据此隔离不同对话上下文
        self.config = {"configurable": {"thread_id": "default"}}

    @property
    def name(self) -> str:
        return f"LangGraph | {self.model}"

    def reply(self, user_input: str) -> str:
        result = self.app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=self.config,
        )
        # 取最后一条 AI 消息作为最终回答
        final_msg = result["messages"][-1]
        return final_msg.content
