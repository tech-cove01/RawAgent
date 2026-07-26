"""
一个极简的 AI Agent，支持工具调用（计算器 + 联网搜索）。

工作流程：
  用户输入 → 模型决策（要调工具吗？）
    ├─ 需要 → 执行工具 → 把结果喂回模型 → 继续决策
    └─ 不需要 → 输出最终回答

依赖任意支持 OpenAI 兼容接口的 LLM
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from tools import TOOLS, execute_tool

# 加载 .env 中的配置
load_dotenv()

# 初始化客户端（OpenAI 兼容接口）
client = OpenAI(
    api_key=os.getenv("GLM_API_KEY"),
    base_url=os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
)
MODEL = os.getenv("GLM_MODEL", "GLM-4.7-Flash")

SYSTEM_PROMPT = """你是一个智能助手，可以调用工具来完成任务。你有以下能力：

1. **calculator** — 当用户需要数学计算时使用（加减乘除、幂运算等）。
2. **web_search** — 当用户询问实时信息、新闻、或你不知道的事实性知识时使用。

规则：
- 尽量直接回答用户的问题，只在确实需要时才调用工具。
- 如果搜索结果不相关，告诉用户你会换一个搜索词再试。
- 用中文回复用户。"""


def main():
    print("=" * 50)
    print(f"  AI Agent 已启动 | 模型: {MODEL}")
    print("  工具: calculator / web_search")
    print("  输入 'exit' 或 'quit' 退出")
    print("=" * 50)

    # 对话历史，system prompt 在最前面
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("再见！")
            break

        # 把用户消息加入对话
        messages.append({"role": "user", "content": user_input})

        # 工具调用循环：反复让模型决定要不要调工具
        while True:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            )
            msg = response.choices[0].message

            # 模型决定不调工具 → 输出回答，结束本次循环
            if not msg.tool_calls:
                print(f"\nAgent> {msg.content}")
                messages.append({"role": "assistant", "content": msg.content})
                break

            # 模型要调工具 → 执行后把结果喂回
            messages.append(msg)  # assistant 消息（含 tool_calls）

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments
                print(f"\n  [调用工具] {tool_name}({tool_args})")

                result = execute_tool(tool_name, tool_args)
                print(f"  [工具结果] {result[:150]}{'...' if len(result) > 150 else ''}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            # 继续循环，让模型基于工具结果再次决策


if __name__ == "__main__":
    main()
