"""
后端②：LangChain AgentExecutor 引擎。
与旧 agent_langchain.py 等价，系统提示词上交给 main.py。
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from tools_langchain import LANGCHAIN_TOOLS

from .base import AgentBackend

load_dotenv()


class LangChainBackend(AgentBackend):
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(llm, LANGCHAIN_TOOLS, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=LANGCHAIN_TOOLS,
            verbose=True,          # 打印内部思考与工具调用，方便学习观察
            handle_parsing_errors=True,
        )

        store = {}

        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        self.agent_with_history = RunnableWithMessageHistory(
            agent_executor,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    @property
    def name(self) -> str:
        return f"LangChain | {self.model}"

    def reply(self, user_input: str) -> str:
        result = self.agent_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}},
        )
        return result["output"]
