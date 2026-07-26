# AI Agent Demo

一个极简的 AI Agent，基于 **OpenAI 兼容接口的 LLM**（默认 GLM-4.7-Flash），支持工具调用。

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
python main.py                      # 默认 raw：纯 OpenAI SDK 手写循环
python main.py --backend langchain  # LangChain 版
python main.py -b langgraph         # LangGraph 版
python main.py -b agents            # OpenAI Agents SDK 版
python main.py -b agents -m GLM-4.7-Flash   # 也可指定模型
```

> 四个框架是「同一个 agent 的四种跑法」，一次对话只能由其中一个引擎驱动；
> 系统提示词和对话主循环只在 `main.py` 维护一份，改一处全局生效。

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

## 项目结构

**单一入口** `main.py` + 四个可切换的后端引擎，实现「同一个 agent（对话 + calculator + web_search）」的四种跑法，方便对照学习：

```
RawAgent/
├── .env.example           # 环境变量模板
├── .gitignore             # 忽略 .env 等敏感文件
├── requirements.txt       # Python 依赖
├── main.py                # 唯一入口：共享 SYSTEM_PROMPT + 对话主循环 + --backend 工厂
├── backends/              # 四个框架引擎（薄插件，各实现 AgentBackend 接口）
│   ├── __init__.py        # 后端工厂 create_backend()
│   ├── base.py            # AgentBackend 统一接口（ABC）
│   ├── raw.py             # 引擎①：纯 OpenAI SDK 手写循环
│   ├── langchain.py       # 引擎②：LangChain AgentExecutor
│   ├── langgraph.py       # 引擎③：LangGraph 状态图
│   └── openai_agents.py   # 引擎④：OpenAI Agents SDK
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
- **对话主循环**：只在 `main.py` 写一份，`backends/` 里每个后端只暴露 `reply(user_input) -> str`。
- **切换引擎**：`python main.py -b <raw|langchain|langgraph|agents>`，再也不用为改提示词去动四个文件。

## 扩展

想加新工具？在 `tools.py` 里：

1. 写一个函数（如 `get_weather(city: str)`）
2. 在 `TOOLS` 列表里加上它的 JSON Schema
3. 在 `execute_tool()` 里加上对应的 `elif` 分支
4. 若要用 LangChain / LangGraph 版，在 `tools_langchain.py` 里也用 `@tool` 包装一份放进 `LANGCHAIN_TOOLS`

## 换模型

只需修改 `.env` 中的三个变量即可，代码不用动：

```env
GLM_API_KEY=你的key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=GLM-4.7-Flash
```

以上为默认配置。想要换其他兼容服务？改 `GLM_BASE_URL` 为对应地址、`GLM_MODEL` 为对应模型名即可，例如 DeepSeek：`GLM_BASE_URL=https://api.deepseek.com/v1`、`GLM_MODEL=deepseek-chat`。
