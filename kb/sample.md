# RawAgent 项目知识库

RawAgent 是一个用于学习和演示的轻量级 AI Agent 框架。它的核心设计目标是：用同一套入口和工具，分别用四种不同的底层引擎把"同一个 agent"跑起来，方便对比和理解各框架的差异。

## 支持的四种后端引擎

RawAgent 目前内置四种后端，通过命令行 `--backend` 选择：

- **raw**：纯 OpenAI SDK 手写的 ReAct 循环，不依赖任何 Agent 框架。这是项目的"主力引擎"，流式输出、RAG 知识库等增强能力都优先在这里实现。
- **langchain**：基于 LangChain 的 Agent 实现，展示主流框架的用法。
- **langgraph**：基于 LangGraph 的状态图实现，展示用图结构编排 Agent 的方式。
- **agents**：基于 OpenAI Agents SDK 的实现（内部使用 Runner.run_sync）。

四个后端都实现统一的 `AgentBackend` 接口（`reply` 与 `stream_reply`），因此 `main.py` 只需面向接口编程，不关心具体是哪个框架在跑。

## 工具

Agent 默认带两个工具：

- **calculator**：安全地计算数学表达式，使用 `ast` 解析而非 `eval`，避免代码注入。
- **web_search**：使用 DuckDuckGo 进行联网搜索，免费免 key，返回前 5 条结果摘要。

工具定义集中在 `tools.py`（OpenAI Function Calling 格式的 `TOOLS` 与 `execute_tool`），是单一来源，各后端共用。

## 记忆与上下文

默认情况下，对话历史保存在后端实例的内存里（`self.messages`），系统提示词在最前面，多轮上下文由各后端自管。重启程序后历史会丢失。RAG 知识库功能会把"外部文档"检索成上下文注入系统提示词，让 agent 能基于文档回答，而不是只靠模型自身知识。

## 如何运行

- 默认运行：`python main.py`
- 指定引擎：`python main.py --backend langchain`
- 开启知识库（需先建库）：`python build_kb.py` 然后 `python main.py --rag`

## 已知限制

- 默认没有持久化记忆，重启即清空。
- RAG 检索基于当前这一轮的用户问题（naive RAG），不做跨轮查询改写。
- 向量库用 chromadb 本地持久化，默认集合名 `kb`，嵌入模型为本地 `BAAI/bge-small-zh-v1.5`，全程离线、免费。
