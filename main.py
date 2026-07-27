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
import asyncio
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


async def main():
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
    parser.add_argument(
        "--rag",
        action="store_true",
        help="启用 RAG 知识库：回答前先检索本地向量库（需先运行 python build_kb.py 建库）",
    )
    parser.add_argument(
        "--kb-dir",
        default="./kb",
        help="知识库文档目录（仅建库时用；运行期不读）",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="向量库持久化目录（与 build_kb.py 的 --persist-dir 对应）",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="关闭 Rerank 重排序（默认开启：向量+BM25 召回后由 cross-encoder 精排）",
    )
    parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="关闭混合检索（默认开启：向量召回 + BM25 关键词召回经 RRF 融合）",
    )
    args = parser.parse_args()

    # 工厂创建后端（系统提示词只在这里注入一次）
    retriever = None
    if args.rag:
        # 惰性导入：未开 --rag 时不加载 fastembed / chromadb，保持轻量启动
        from rag import VectorStore
        if not os.path.exists(args.persist_dir):
            print(f"  [警告] 未找到知识库目录 {args.persist_dir}")
            print(f"         请先运行: python build_kb.py {args.kb_dir}")
            print("         本次将不带知识库运行。")
        else:
            retriever = VectorStore(
                args.persist_dir,
                rerank=not args.no_rerank,
                hybrid=not args.no_hybrid,
            )
    backend = create_backend(args.backend, args.model, SYSTEM_PROMPT, retriever=retriever)

    print("=" * 50)
    print(f"  AI Agent 已启动 | 引擎: {backend.name}")
    print("  工具: calculator / web_search")
    print(f"  知识库: {'开启 (RAG)' if retriever else '关闭'}")
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

        # 流式输出：逐字打印 stream_reply 产出的文本片段
        print("\nAgent> ", end="", flush=True)
        async for delta in backend.stream_reply(user_input):
            print(delta, end="", flush=True)
        print()  # 一轮结束补换行


if __name__ == "__main__":
    asyncio.run(main())
