"""
后端④：OpenAI Agents SDK 引擎（轻量框架，Runner 跑循环）。
与旧 agent_openai_agents.py 等价，系统提示词上交给 main.py。

注意：OpenAI Agents SDK 默认走 Responses API，只有 OpenAI 自家支持。
用智谱等 OpenAI 兼容接口需：
  1. set_default_openai_client 指向兼容 base_url
  2. set_default_openai_api("chat_completions") 切到 chat/completions
  3. set_tracing_disabled(True) 关掉默认 trace 上报
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

from tools import calculator, web_search

from .base import AgentBackend

load_dotenv()


class OpenAIAgentsBackend(AgentBackend):
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        client = AsyncOpenAI(
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        )
        set_default_openai_client(client)
        set_default_openai_api("chat_completions")  # 兼容接口只支持 chat/completions
        set_tracing_disabled(True)                  # 兼容接口无 trace 端点

        # 直接把 tools.py 的框架无关函数包成 function_tool
        tools = [function_tool(calculator), function_tool(web_search)]
        self.agent = Agent(
            name="assistant",
            model=model,
            instructions=system_prompt,  # 相当于 system prompt
            tools=tools,
        )
        # 多轮上下文：每轮把上一轮完整输入列表作为下一轮输入
        self.history = []

    @property
    def name(self) -> str:
        return f"OpenAI-Agents | {self.model}"

    def reply(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        result = Runner.run_sync(self.agent, self.history)
        self.history = result.to_input_list()
        return result.final_output
