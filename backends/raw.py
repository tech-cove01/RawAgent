"""
后端①：纯 OpenAI SDK 手写循环（自研薄引擎）。
与旧 agent.py 等价，只是把「系统提示词 + 主循环」上交给 main.py，
本类只保留「引擎 glue 代码」。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, execute_tool

from .base import AgentBackend

load_dotenv()


class RawBackend(AgentBackend):
    def __init__(self, model: str, system_prompt: str):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        )
        # 对话历史：system prompt 在最前面，多轮上下文自管
        self.messages = [{"role": "system", "content": system_prompt}]

    @property
    def name(self) -> str:
        return f"raw (OpenAI SDK) | {self.model}"

    def reply(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        # 工具调用循环：反复让模型决定要不要调工具
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message

            # 模型决定不调工具 → 输出回答，结束本次循环
            if not msg.tool_calls:
                content = msg.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content

            # 模型要调工具 → 执行后把结果喂回，继续循环
            self.messages.append(msg)  # assistant 消息（含 tool_calls）

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments
                print(f"\n  [调用工具] {tool_name}({tool_args})")

                result = execute_tool(tool_name, tool_args)
                print(f"  [工具结果] {result[:150]}{'...' if len(result) > 150 else ''}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
