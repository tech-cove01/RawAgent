# AI Agent Demo

一个在拓展中的 AI Agent，基于 **OpenAI 兼容接口的 LLM**（默认 GLM-4.7-Flash），支持工具调用。

## 能力

- 智能对话
- 安全数学计算（calculator）
- 联网搜索（web_search，通过 DuckDuckGo，免费免 key）

## 快速开始

### 1. 获取 API Key

1. 准备一个支持 OpenAI 兼容接口的 LLM 服务的 API Key（如智谱、DeepSeek 等均可）
2. 进入对应平台的「API Keys」页面，创建一个新 key 并复制

### 2. 配置环境变量

```bash
# 复制模板文件
copy .env.example .env

# 用编辑器打开 .env，把 your_api_key_here 替换成你真实的 key
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行

**单入口**，通过 `--backend`（或 `-b`）选择用哪个框架引擎驱动同一个 agent：

```bash
python main.py                      # 默认 raw：纯 OpenAI SDK 手写循环（逐字流式）
python main.py --backend langchain  # LangChain 版
python main.py -b langgraph         # LangGraph 版
python main.py -b agents            # OpenAI Agents SDK 版
python main.py -b agents -m GLM-4.7-Flash   # 也可指定模型
```

> 四个框架是「同一个 agent 的四种跑法」，一次对话只能由其中一个引擎驱动；
> 系统提示词和对话主循环只在 `main.py` 维护一份，改一处全局生效。
>
> **输出为逐字流式**：`main.py` 主循环异步驱动 `backend.stream_reply()`，边生成边打印。
> 其中只有 **raw** 引擎实现了真正的逐字流式；其余三个后端（`langchain` / `langgraph` / `agents`）
> 作为「用过这些框架」的实现证据保留在仓库中，经基类默认回退一次性吐出（仍可用、仍演示框架）。

## 使用示例

```
你> 123 * 456 等于多少？

  [调用工具] calculator("123*456")
  [工具结果] 56088

Agent> 123 × 456 = 56088

你> 今天美国股市大盘怎么样？

  [调用工具] web_search("美国股市 2026年7月 大盘行情")

Agent> 根据搜索结果，今天美股三大指数...
```

> 上面的 `Agent>` 在实际运行中是**逐字流式**打印的（raw 引擎）。
> 若当前环境未安装对应框架或版本不兼容，`-b langchain/langgraph/agents` 会在选中时才尝试加载对应模块，
> 因此**默认的 raw 引擎完全不受其他框架缺失的影响**，可独立启动。

## 项目结构

**单一入口** `main.py` + 四个可切换的后端引擎，实现「同一个 agent（对话 + calculator + web_search）」的四种跑法，方便对照学习：

```
RawAgent/
├── .env.example           # 环境变量模板
├── .gitignore             # 忽略 .env 等敏感文件
├── requirements.txt       # Python 依赖
├── main.py                # 唯一入口：共享 SYSTEM_PROMPT + 对话主循环 + --backend 工厂
├── backends/              # 四个框架引擎（薄插件，各实现 AgentBackend 接口）
│   ├── __init__.py        # 后端工厂 create_backend()（惰性导入，避免框架缺失拖累启动）
│   ├── base.py            # AgentBackend 统一接口（ABC）：reply() + 带默认回退的 stream_reply()
│   ├── raw.py             # 引擎①【主维护】：纯 OpenAI SDK 手写循环 + 逐字流式 + 限流重试
│   ├── langchain.py       # 引擎②（演示保留）：LangChain AgentExecutor
│   ├── langgraph.py       # 引擎③（演示保留）：LangGraph 状态图
│   └── openai_agents.py   # 引擎④（演示保留）：OpenAI Agents SDK
├── tools.py               # 工具函数（框架无关）+ OpenAI Function Calling 的 JSON Schema，单一来源
├── tools_langchain.py     # LangChain @tool 包装（LangChain / LangGraph 版共用的单一来源）
└── README.md              # 本文件
```

运行（功能完全一致，只是引擎不同）：

```bash
python main.py                      # 默认 raw 版
python main.py --backend langchain  # LangChain 版
python main.py -b langgraph         # LangGraph 版
python main.py -b agents            # OpenAI Agents SDK 版
```

### 工具定义的「单一来源」

`tools.py` 集中维护工具逻辑，避免重复：

- `calculator()` / `web_search()`：纯逻辑函数，框架无关（四个后端共用）
- `TOOLS` + `execute_tool()`：OpenAI Function Calling 的 JSON Schema，给 raw 后端用
- `tools_langchain.py` 里的 `LANGCHAIN_TOOLS`：用 `@tool` 包装的 LangChain 工具列表，给 LangChain / LangGraph 后端共用

这样改工具描述只改一处，不会「改了一边忘改另一边」。且 `tools.py` 保持框架无关（raw 后端只需 `openai` + `ddgs` 即可运行，`tools_langchain.py` 才引入 langchain 依赖）。

## 单入口说明（为什么只有一个入口）

四个框架是「同一个 agent 的四种跑法」，不是互补的层次，所以一次对话只能由其中一个引擎驱动。
重构后：

- **系统提示词**：只在 `main.py` 的 `SYSTEM_PROMPT` 维护一份，改一处全局生效。
- **对话主循环**：只在 `main.py` 写一份（异步循环），统一调用 `backend.stream_reply(user_input)` 逐字打印；
  `backends/` 里每个后端实现 `reply(user_input) -> str`，并提供 `stream_reply()` 流式接口
  （基类有默认回退，只有 **raw** 实现了真正的逐字流式）。
- **切换引擎**：`python main.py -b <raw|langchain|langgraph|agents>`，再也不用为改提示词去动四个文件。
- **惰性导入**：`create_backend` 在工厂内惰性加载对应模块，除 raw 外的引擎只在被显式选中时才 `import`，
  因此其他框架缺失/版本不兼容时，默认的 raw 引擎仍可独立启动。

## 扩展

想加新工具？在 `tools.py` 里：

1. 写一个函数（如 `get_weather(city: str)`）
2. 在 `TOOLS` 列表里加上它的 JSON Schema
3. 在 `execute_tool()` 里加上对应的 `elif` 分支
4. 若要用 LangChain / LangGraph 版，在 `tools_langchain.py` 里也用 `@tool` 包装一份放进 `LANGCHAIN_TOOLS`

> 后续进阶能力（Milvus 持久记忆、规划层、自我反思等）都建议在 **raw** 引擎的
> `stream_reply()` 工具循环里扩展——它是唯一持续维护的引擎，逻辑全摊在明处、最好改。

## 限流重试（Rate Limit）

免费模型（如智谱 GLM 免费档）高峰期可能返回 `429`（错误码 1305「访问量过大」）。
`raw.py` 的 `_create_with_retry()` 实现了**指数退避重试**（默认最多 3 次，间隔 `2^attempt` 秒），
在 `reply()` 与 `stream_reply()` 调用模型前自动生效；遇到瞬时限流会在终端打印
`[限流] 429，Ns 后重试 (x/3)` 并重试，而非直接崩溃。

## 换模型 / 换平台

只需修改 `.env` 中的三个变量即可，代码不用动：

```env
GLM_API_KEY=你的key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=GLM-4.7-Flash
```

以上为默认配置。因为 `raw.py` 是纯 OpenAI SDK，任意兼容 OpenAI 接口的平台都能直接换：

| 平台 | `GLM_BASE_URL` | `GLM_MODEL` 示例 |
|------|----------------|------------------|
| 智谱（默认） | `https://open.bigmodel.cn/api/paas/v4` | `GLM-4.7-Flash` |
| 硅基流动 SiliconFlow | `https://api.siliconflow.cn/v1` | `Qwen/Qwen3-8B-Instruct` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 通义千问 DashScope | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Kimi 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

> 频繁 429 时优先换 SiliconFlow / DeepSeek 等容量更足的免费平台，或错峰重试。

## RAG 知识库（向量检索）

RawAgent 的 **raw** 引擎支持接一个本地向量知识库：把外部文档切块、本地嵌入、存入 chromadb，
回答前先检索 top-k 片段注入系统提示词，让 agent「有据可依」地回答，而不是只靠模型自身知识。

嵌入用本地模型 `BAAI/bge-small-zh-v1.5`（fastembed 驱动），**全程离线、免费、不耗 key**；
向量库用 chromadb 本地持久化（默认 `./chroma_db`，底层 HNSW、支持元数据过滤）。
langchain / langgraph / agents 三个后端不接知识库，保持演示价值。

### 建库

```bash
python build_kb.py                  # 默认读 ./kb，写 ./chroma_db
python build_kb.py ./kb --reset     # 清空旧库后重建
```

把你的 `.md` / `.txt` 放进 `kb/` 目录即可被收录。切块策略（大小 / overlap）在 `rag.py` 的
`CHUNK_SIZE` / `CHUNK_OVERLAP` 调整。

### 运行（带知识库）

```bash
python main.py --rag                 # raw 引擎 + 知识库
```

未建库就开 `--rag` 会打印警告并降级为「无知识库」运行，不会崩溃。
检索命中时终端会打印 `[检索] 命中 3 段`，随后是基于知识库给出的回答。

> RAG 相关代码集中在 `rag.py`（切块 / 嵌入 / 向量库）与 `build_kb.py`（建库脚本），
> 检索注入逻辑只在 `backends/raw.py` 的 `_inject_knowledge()` 一处，便于阅读与扩展。

### 检索阈值门控（减少噪声）

RAG 采用「相似度阈值门控」：每轮仍会做本地检索（开销极小），但只有 top-1 相似度
≥ 阈值才把知识库上下文注入 prompt；闲聊、追问等不相关问题会自动跳过，保持 prompt 干净。

- 阈值在 `backends/raw.py` 的 `RawBackend.rag_threshold`（默认 `0.5`）调整。
- 知识库内容密集、想更宽松命中 → 调低（如 `0.4`）；想更严格过滤噪声 → 调高（如 `0.6`）。
- 终端日志可观测每次检索结果：命中显示 `[检索] 命中 3 段 (相似度 0.xx)`，
  跳过显示 `[检索] 跳过 (相似度 0.xx < 阈值 0.5)`，方便据此调参。
- 相似度由 chromadb 的余弦距离换算：`sim = 1 - distance`（0=不相似，1=相同）。

### 检索增强：混合检索 + Rerank 重排序（默认开启）

默认检索链路在纯向量召回之外加了两步业界标准优化，命中质量与抗噪都更好：

1. **混合检索（向量 + BM25 关键词）**：向量召回擅长语义，BM25 擅长「精确关键词」（如型号、专有名词、报错码）；
   两路各取 top-20 候选，用 **RRF（Reciprocal Rank Fusion）** 融合成统一候选池，解决「余弦分」和「BM25 分」量纲不可比、
   无法直接相加的问题。中文关键词按**字**切分（零额外依赖），若环境装了 `jieba` 会自动改用分词、召回更精准。
2. **Rerank 重排序**：融合后的候选交给 cross-encoder（`BAAI/bge-reranker-base`，fastembed 驱动）逐条打分精排，
   取 top-k 注入 prompt。cross-encoder 同时看到「问题+片段」，比单纯向量内积更能判断真实相关度
   （实测能给不相关片段打出负分、相关片段打出高分，过滤噪声更干净）。

开关（默认都开，想退化可关）：

```bash
python main.py --rag                 # 默认：向量 + BM25(RRF) → Rerank
python main.py --rag --no-rerank     # 关 Rerank，只保留向量 + BM25 混合召回
python main.py --rag --no-hybrid     # 关混合检索，纯向量召回（行为等价于早期版本）
```

> Rerank 模型权重在**首次**运行时由 fastembed 自动下载（ONNX，约百 MB 级），之后本地缓存、离线可用。
>
> 注：本机 fastembed 0.8 的 cross-encoder 支持列表里**没有** `bge-reranker-v2-m3`，代码改用同源中文友好的
> `BAAI/bge-reranker-base`；如需其它重排模型（如 `jinaai/jina-reranker-v2-base-multilingual`），
> 改 `rag.py` 的 `RERANK_MODEL` 常量即可。

**门控语义保持不变**：注入与否仍由 top-1 的相关度分数 ≥ `rag_threshold`（默认 0.5）决定——
Rerank / 混合检索只改变「排序与候选来源」，不改变阈值含义。终端日志同步展示当前管线，方便调参：

```
  [检索] 命中 3 段 (管线:向量+BM25(RRF)→Rerank 门控分 0.xx)

