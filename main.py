"""
AI Agent 统一入口（单入口）。

四个框架（raw / LangChain / LangGraph / OpenAI-Agents）是「同一个 agent 的四种跑法」，
本文件是它们的唯一入口：
  - 系统提示词 SYSTEM_PROMPT 只维护这一份（改一处全局生效）
  - 对话主循环只写这一份
  - 具体用哪个框架驱动，由 --backend 指定（默认 raw）

运行示例：
  python main.py                      # 默认 raw 手写循环
  python main.py --backend langchain  # LangChain 引擎
  python main.py -b langgraph         # LangGraph 引擎
  python main.py -b agents -m GLM-4.7-Flash
"""

import argparse
import os

from dotenv import load_dotenv

from backends import create_backend

load_dotenv()

# ============================================================
# 系统提示词（单一来源）：四个后端共用，改这里即全局生效
# ============================================================
SYSTEM_PROMPT = """你是一个智能助手，可以调用工具来完成任务。你有以下能力：

1. **calculator** — 当用户需要数学计算时使用（加减乘除、幂运算等）。
2. **web_search** — 当用户询问实时信息、新闻、或你不知道的事实性知识时使用。

规则：
- 尽量直接回答用户的问题，只在确实需要时才调用工具。
- 如果搜索结果不相关，告诉用户你会换一个搜索词再试。
- 用中文回复用户。"""


def main():
    parser = argparse.ArgumentParser(
        description="AI Agent 单入口：用不同框架引擎驱动同一个 agent。"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["raw", "langchain", "langgraph", "agents"],
        default="raw",
        help="选择驱动 agent 的框架引擎（默认 raw）",
    )
    parser.add_argument(
        "--model", "-m",
        default=os.getenv("GLM_MODEL", "GLM-4.7-Flash"),
        help="模型名（默认读 GLM_MODEL 环境变量或 GLM-4.7-Flash）",
    )
    args = parser.parse_args()

    # 工厂创建后端（系统提示词只在这里注入一次）
    backend = create_backend(args.backend, args.model, SYSTEM_PROMPT)

    print("=" * 50)
    print(f"  AI Agent 已启动 | 引擎: {backend.name}")
    print("  工具: calculator / web_search")
    print("  输入 'exit' 或 'quit' 退出")
    print("=" * 50)

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

        answer = backend.reply(user_input)
        print(f"\nAgent> {answer}")


if __name__ == "__main__":
    main()
