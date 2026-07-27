"""
后端①：纯 OpenAI SDK 手写循环（自研薄引擎）。
与旧 agent.py 等价，只是把「系统提示词 + 主循环」上交给 main.py，
本类只保留「引擎 glue 代码」。
"""

import os
import time
from typing import AsyncIterator

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError

from tools import TOOLS, execute_tool

from .base import AgentBackend

load_dotenv()


class RawBackend(AgentBackend):
    def __init__(self, model: str, system_prompt: str, retriever=None):
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("GLM_API_KEY"),
            base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        )
        # 原始系统提示词（不含 RAG 注入），每轮检索后据此重建 messages[0]
        self.system_prompt = system_prompt
        # 对话历史：system prompt 在最前面，多轮上下文自管
        self.messages = [{"role": "system", "content": system_prompt}]
        # 可选检索器：RAG 知识库（仅 raw 引擎接入，其余后端不传）
        self.retriever = retriever
        self.rag_k = 3
        # RAG 注入阈值：top-1 相似度 ≥ 此值才把知识库上下文注入 prompt。
        # 低于阈值视为与知识库无关，自动跳过（闲聊/追问不再被知识库绑架）。
        # 知识库密度高可调低（如 0.4），稀疏可调高（如 0.6）以更严格过滤噪声。
        self.rag_threshold = 0.5
        # 限流重试：免费模型高峰期易 429，指数退避后自动重试
        self.max_retries = 3
        self.retry_base = 2

    def _inject_knowledge(self, user_input: str) -> None:
        """RAG 阈值门控：每轮检索 top-k，但仅当 top-1 相似度 ≥ 阈值才注入知识库上下文。

        每轮都从原始 system_prompt 重建 messages[0]，避免多轮对话里检索片段在
        messages[0] 累积重复；不相关的问题（相似度低于阈值）自动跳过，保持 prompt 干净。
        """
        if not self.retriever:
            return
        ctx, sim, pipeline = self.retriever.retrieve(user_input, k=self.rag_k)
        if ctx and sim >= self.rag_threshold:
            self.messages[0]["content"] = (
                self.system_prompt + "\n\n[知识库上下文]\n" + ctx
            )
            print(f"\n  [检索] 命中 {self.rag_k} 段 (管线:{pipeline} 门控分 {sim:.2f})", flush=True)
        else:
            # 没命中或相似度不足都要重置回原始提示词，避免沿用上一轮的旧上下文
            self.messages[0]["content"] = self.system_prompt
            if ctx:
                print(
                    f"\n  [检索] 跳过 (管线:{pipeline} 门控分 {sim:.2f} < 阈值 {self.rag_threshold})",
                    flush=True,
                )

    @property
    def name(self) -> str:
        return f"raw (OpenAI SDK) | {self.model}"

    def _create_with_retry(self, stream: bool = False):
        """带指数退避重试地调用模型，缓解 429 限流（如免费模型高峰期过载）。"""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    stream=stream,
                )
            except RateLimitError as err:
                last_err = err
                if attempt >= self.max_retries:
                    break
                wait = self.retry_base ** attempt
                print(f"\n  [限流] 429，{wait}s 后重试 ({attempt}/{self.max_retries})", flush=True)
                time.sleep(wait)
        # 走到这里说明已耗尽重试；last_err 必为捕获到的 RateLimitError
        assert last_err is not None
        raise last_err

    def reply(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._inject_knowledge(user_input)

        # 工具调用循环：反复让模型决定要不要调工具
        while True:
            response = self._create_with_retry(stream=False)
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

    async def stream_reply(self, user_input: str) -> AsyncIterator[str]:
        self.messages.append({"role": "user", "content": user_input})
        self._inject_knowledge(user_input)

        # 工具调用循环：反复让模型决定要不要调工具
        while True:
            stream = self._create_with_retry(stream=True)

            # 流式期间累积：正文片段 + 跨 chunk 拼接的 tool_calls
            content_parts = []
            # tool_calls 按索引累积：每个槽位 {id, name, arguments}
            tool_calls = {}

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # 正文流式输出
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content

                # 工具调用片段：跨 chunk 按 index 累积
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc_delta.id:
                            slot["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            slot["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            slot["arguments"] += tc_delta.function.arguments

            # 整个回答没有工具调用 → 收尾，把正文存入上下文
            if not tool_calls:
                content = "".join(content_parts)
                self.messages.append({"role": "assistant", "content": content})
                return

            # 模型要调工具 → 执行后把结果喂回，继续循环（下一轮可能又有正文）
            accumulated = [tool_calls[i] for i in sorted(tool_calls.keys())]
            self.messages.append({
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [
                    {
                        "id": t["id"],
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "arguments": t["arguments"],
                        },
                    }
                    for t in accumulated
                ],
            })

            for t in accumulated:
                tool_name = t["name"]
                tool_args = t["arguments"]
                print(f"\n  [调用工具] {tool_name}({tool_args})", flush=True)

                result = execute_tool(tool_name, tool_args)
                print(f"  [工具结果] {result[:150]}{'...' if len(result) > 150 else ''}", flush=True)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": t["id"],
                    "content": result,
                })
