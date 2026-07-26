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

```bash
python agent.py
```

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

```
demo/
├── .env.example    # 环境变量模板
├── .gitignore      # 忽略 .env 等敏感文件
├── requirements.txt # Python 依赖
├── tools.py        # 工具函数 + Schema 定义
├── agent.py        # Agent 主循环
└── README.md       # 本文件
```

## 扩展

想加新工具？在 `tools.py` 里：

1. 写一个函数（如 `get_weather(city: str)`）
2. 在 `TOOLS` 列表里加上它的 JSON Schema
3. 在 `execute_tool()` 里加上对应的 `elif` 分支

## 换模型

只需修改 `.env` 中的三个变量即可，代码不用动：

```env
GLM_API_KEY=你的key
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_MODEL=GLM-4.7-Flash
```

以上为默认配置。想要换其他兼容服务？改 `GLM_BASE_URL` 为对应地址、`GLM_MODEL` 为对应模型名即可，例如 DeepSeek：`GLM_BASE_URL=https://api.deepseek.com/v1`、`GLM_MODEL=deepseek-chat`。
