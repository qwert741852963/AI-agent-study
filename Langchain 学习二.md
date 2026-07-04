## 一、工具tool
### 1.1 @toll

**✍️ 基本用法：从函数到工具**
```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索匹配查询的记录。"""
    return f"为 '{query}' 找到了 {limit} 条结果"
```

**🎨 进阶定制：名称与描述**

默认情况下，工具名就是函数名，描述就是文档字符串。你也可以在装饰器中显式地覆盖它们，以便更精确地引导模型。

```python
# 自定义名称
@tool("web_search")
def search(query: str) -> str:
    """在网络上搜索信息。"""
    return f"'{query}' 的搜索结果"

# 同时自定义名称和描述
@tool("calculator", description="执行算术计算。遇到任何数学问题时请使用此工具。")
def calc(expression: str) -> str:
    """计算数学表达式。"""
    return str(eval(expression))
```

**复杂参数：使用 Pydantic 定义输入模式**

当工具需要更复杂的输入（如参数校验、枚举值、多字段对象）时，可以通过 `args_schema` 参数传入一个 Pydantic `BaseModel` 来明确定义。

```python
from pydantic import BaseModel, Field
from typing import Literal

# 1. 定义输入模型
class WeatherInput(BaseModel):
    location: str = Field(description="城市名称或坐标")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius", description="温度单位偏好"
    )

# 2. 在 @tool 中指定 args_schema
@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius") -> str:
    """获取当前天气。"""
    # 工具内部的参数名必须与模型定义的一致
    return f"{location}的天气是晴朗的，{units}."
```

**重要参数**

`@tool` 装饰器还提供了其他参数，用于控制工具的行为。

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `name` | `str` | 函数名 | 自定义工具名称。 |
| `description` | `str` | 文档字符串 | 自定义工具描述。 |
| `args_schema` | `BaseModel` | `None` | 使用 Pydantic 模型定义更严格的输入模式。 |
| `return_direct` | `bool` | `False` | 如果设为 `True`，工具的输出会直接返回给用户，而不会再次传递给模型进行后续处理。 |

**Agent 中使用**

定义好工具后，将其放入一个列表，并在创建智能体（Agent）时传入即可。

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 假设已定义好 multiply 和 divide 工具
tools = [multiply, divide]
model = init_chat_model(model="gpt-4o", model_provider="openai")

# 创建 Agent
agent = create_agent(model=model, tools=tools)

# 运行 Agent
result = agent.invoke({"messages": [{"role": "user", "content": "请帮我计算 (8 * 3) / 2 的结果"}]})
print(result)
```

### 1.2 BaseTool

`BaseTool` 是 **LangChain** 框架中的一个核心抽象基类。你可以把它理解为所有“工具”的“模板”或“蓝图”。通过继承这个基类并实现其中的方法，你可以为大型语言模型（LLM）创建拥有高度自定义能力的工具。

**为什么需要 BaseTool？**

在 LangChain 中，主要有三种创建工具的方式，它们的对比能帮你理解 `BaseTool` 的定位：

| 创建方式 | 易用性 | 灵活性 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **`@tool` 装饰器** | ⭐⭐⭐⭐⭐ (最简单) | ⭐⭐⭐ (基础功能) | 快速将简单函数转为工具 |
| **`StructuredTool.from_function`** | ⭐⭐⭐⭐ (灵活配置) | ⭐⭐⭐⭐ (支持异步等) | 需要更多配置，如同步/异步方法 |
| **继承 `BaseTool` 类** | ⭐⭐ (最复杂) | ⭐⭐⭐⭐⭐ (完全控制) | **需要高度自定义、维护内部状态或依赖注入的复杂场景** |

简单来说，当 `@tool` 装饰器无法满足你的复杂需求时，`BaseTool` 就是那个提供最大自由度的选择。

**BaseTool 的核心要素**

创建一个 `BaseTool` 的子类，需要定义以下几个关键部分：

1.  **`name` (名称)**：工具的唯一标识符。
2.  **`description` (描述)**：**最重要的部分**，LLM 根据它来决定何时使用该工具。
3.  **`args_schema` (参数模式)**：一个 Pydantic `BaseModel` 子类，用于定义和验证工具的输入参数。
4.  **`_run()` (核心逻辑)**：**必须实现**的方法，包含工具被调用时的同步执行逻辑。
5.  **`_arun()` (异步逻辑)**：可选，用于实现异步执行。
6.  **`return_direct` (返回策略)**：可选，若设为 `True`，工具的输出会直接返回给用户，而不会继续传给 LLM。

**代码示例**

下面是一个使用 `BaseTool` 创建乘法计算器的完整示例。

```python
# 1. 导入必要的库
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# 2. 定义输入参数的模型 (Pydantic)
class CalculatorInput(BaseModel):
    a: int = Field(description="第一个数字")
    b: int = Field(description="第二个数字")

# 3. 继承 BaseTool 并实现自定义工具
class CustomCalculatorTool(BaseTool):
    # 定义工具的基本信息
    name = "Calculator"
    description = "用于乘法计算" 
    args_schema = CalculatorInput 
    return_direct = True # 结果直接返回给用户

    # 实现核心逻辑
    def _run(self, a: int, b: int) -> int:
        """执行工具的逻辑：乘法运算"""
        return a * b

# 4. 使用工具
tool = CustomCalculatorTool()
result = tool.invoke({"a": 2, "b": 3})
print(result) # 输出: 6
```

**注意事项**

*   使用 `BaseTool` 需要 `langchain-core>=0.2.14`。
*   直接继承 `BaseTool` 的代码量比 `@tool` 装饰器多，适用于对工具行为有**完全控制权**的复杂场景。
*   如果工具逻辑不复杂，LangChain 官方更推荐使用 `@tool` 装饰器。

### 1.3 langchain的内置工具

LangChain 本身是一个框架，其“内置工具”主要可以分为两类：一是**框架官方集成并维护的工具**（主要在 `langchain_community.tools` 中），二是**由模型提供商提供的、在云端执行的工具**。

**LangChain 社区集成工具**

这是指通过 `langchain_community.tools` 模块提供的大量预置工具。这些工具覆盖面很广，可按功能分为以下几类：

| 类别 | 工具示例 | 主要功能 |
| :--- | :--- | :--- |
| **🔍 搜索与信息检索** | `DuckDuckGoSearchRun`, `GoogleSearchRun`, `WikipediaQueryRun`, `ArXivQueryRun`, `PubMedQueryRun` | 执行网络搜索、查询维基百科、检索学术论文或医学文献等。 |
| **💻 代码执行与计算** | `PythonREPLTool`, `CalculatorTool` | 执行Python代码片段，或进行数学计算。 |
| **🗄️ 数据库与SQL** | `QuerySqlTool`, `ListTablesSqlTool` | 查询SQL数据库、列出数据库表等。 |
| **📁 文件系统操作** | `FileSystemTool` | 执行文件的读写、创建、删除、列出等操作。 |
| **📄 文档与数据处理** | `PDFMinerTool`, `CSVTool` | 读取PDF、CSV等格式的文档内容并进行处理。 |
| **🌐 API与网络请求** | `RequestsGetTool`, `RequestsPostTool` | 发送HTTP GET/POST请求，与外部API交互。 |
| **🎬 多媒体与杂项** | `YouTubeSearchTool`, `TranslateTool` | 搜索YouTube视频、进行多语言翻译等。 |

> **请注意**：在 LangChain v0.1.0 版本之后，大部分内置工具已从 `langchain.tools` 迁移到了 `langchain_community.tools`。建议使用新版的导入路径。

**模型提供商内置工具**

一些模型提供商（如OpenAI、Anthropic）会在其模型服务中直接提供预配置的工具。这些工具通常在**模型服务端执行**，无需用户自己搭建和维护。典型的例子包括**网络搜索**和**代码解释器**。

**如何进一步探索**

官方文档和集成目录是获取完整列表的最佳途径。此外，LangChain还提供了**工具包（Toolkits）**，即针对特定场景（如操作SQL数据库）的一组工具集合。

如果想了解某一类具体工具的用法，可以随时再问我。

### 1.4 Toolkits工具包
**Toolkits（工具包）是 LangChain 中一组旨在协同工作、解决特定领域问题的工具的集合**。

如果说单个 `Tool` 是让 AI 拥有“动手能力”的“手”，那么 `Toolkit` 就是一套为完成某个特定任务而预装配好的“工具箱”或“瑞士军刀”。

**核心价值：解决复杂问题**

*   **逻辑分组与资源共用**：Toolkit 能将多个工具围绕一个共同的资源进行逻辑分组和初始化。例如，`SQLDatabaseToolkit` 的多个工具可以共用同一个数据库连接。这样，创建工具包时只需初始化一次资源，所有工具即可共享使用，避免了重复连接。
*   **简化配置与使用**：Toolkit 提供了 `get_tools()` 方法，可以一次性获取该工具包中所有预配置好的工具列表，便于批量绑定给智能体（Agent），无需逐个手动创建和配置。
*   **聚焦特定用例**：Toolkit 的设计初衷是为特定用例创建专门的智能体。例如，需要处理自然语言与数据库交互的问题时，可以直接使用 `SQLDatabaseToolkit` 及其配套的智能体。

**常见的内置工具包 (Toolkits)**

LangChain 提供了丰富的内置工具包，覆盖多种常见场景。以下是一些典型的例子：

| 工具包 (Toolkit) | 用途 | 核心资源/场景 |
| :--- | :--- | :--- |
| **SQLDatabaseToolkit** | 让智能体与SQL数据库交互，执行查询、获取表信息等。 | 数据库连接 (SQLDatabase) |
| **VectorStoreToolkit** | 将向量数据库（Vector Store）封装成一个工具，用于语义搜索和问答。 | 向量数据库 (VectorStore) |
| **OpenAPIToolkit** | 让智能体能够理解和调用符合OpenAPI（Swagger）规范的API。 | OpenAPI规范文件 |
| **FileSystemToolkit** | 提供与本地文件系统交互的工具，如读写文件、列出目录等。 | 本地文件系统 |
| **JSONToolkit** | 用于处理和分析大型JSON数据，解决其超出上下文窗口的问题。 | JSON对象 |
| **RequestsToolkit** | 提供工具来发送HTTP请求（GET, POST等），用于与Web API交互。 | HTTP请求 |
| **PowerBIToolkit** | 允许智能体与Power BI数据集交互和查询。 | Power BI数据集 |

除了这些，还有用于处理特定格式的 `NLAToolkit` (自然语言API)、`SparkSQLToolkit`等。

**如何使用一个工具包**

使用工具包的流程非常标准化，通常分为三步：

1.  **实例化工具包**：创建工具包实例时，需要传入其依赖的核心资源（如数据库连接、向量存储实例等）。
2.  **获取工具**：调用工具包的 `get_tools()` 方法，获取预配置好的工具列表。
3.  **创建智能体**：将这些工具传递给智能体（Agent），让它能够使用这些工具来完成任务。

**代码示例 (SQLDatabaseToolkit)**：

```python
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain import hub

# 1. 初始化核心资源：数据库连接和LLM
db = SQLDatabase.from_uri("sqlite:///Chinook.db") # 示例数据库[reference:1][reference:2]
llm = ChatOpenAI(temperature=0)

# 2. 实例化工具包[reference:3][reference:4]
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# 3. 获取工具列表
tools = toolkit.get_tools() # 包含查询、检查表结构等工具[reference:5]

# (可选) 为智能体准备一个系统提示词，指定数据库方言等[reference:6]
prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
system_message = prompt_template.format(dialect="SQLite", top_k=5)

# 4. 创建并运行智能体[reference:7]
agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
example_query = "哪个国家的客户消费最多？" # Which country's customers spent the most?[reference:8][reference:9]
events = agent_executor.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```

## 二、Agent智能体
### 2.1 介绍
**一个完整的 Agent 由以下四大核心组件构成**：

| 组件 | 作用 | 关键说明 |
| :--- | :--- | :--- |
| **模型 (Model)** | **推理引擎** | 负责理解任务、进行推理、制定计划并决定调用哪些工具。 |
| **工具 (Tools)** | **能力扩展** | Agent 的“手脚”，是它能与外部世界（API、数据库、文件系统等）交互的接口。 |
| **系统提示 (System Prompt)** | **行为准则** | 用于设定 Agent 的角色、目标和行为边界，是引导模型有效推理的关键。 |
| **中间件 (Middleware)** | **行为增强** | LangChain 1.0 的核心创新，允许开发者以可组合的方式插入自定义逻辑，如日志、速率限制、动态上下文注入等。 |

---
**`create_agent` 的核心参数**

`create_agent` 提供了丰富的配置选项，让你能精细控制 Agent 的行为。

*   **`model`**: 指定 Agent 使用的模型。
*   **`tools`**: 传入工具列表。
*   **`system_prompt`**: 设定系统级指令。
*   **`response_format`**: 定义结构化输出。
*   **`context_schema`**: 定义传递给工具的上下文数据结构。
*   **`middleware`**: 插入中间件以扩展功能。


### 2.2 最简创建与调用

```python
from langchain.agents import create_agent
from langchain.tools import tool

# 1. 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的当前天气"""
    return f"{city} 的天气是晴朗的，温度25°C。"

# 2. 创建 Agent
agent = create_agent(
    model="openai:gpt-4o",  # 推荐使用 "提供商:模型名" 的格式
    tools=[get_weather],
)

# 3. 调用 Agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "北京的天气怎么样？"}]}
)
print(result["messages"][-1].content)
```

### 2.3 状态管理与调用

Agent 的状态（State）本质上是**消息列表**。通过 `thread_id` 可以实现多轮对话的上下文管理。

```python
from langchain_core.utils.uuid import uuid6

thread_id = str(uuid6())

config = {"configurable": {"thread_id": thread_id}}

# 第一轮
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明。"}]},
    config=config
)
# 第二轮，Agent 能记住你的名字
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
```

### 2.4 Stream 流式输出

流式输出能显著提升用户体验。`agent.stream()` 方法支持多种模式。

```python
# 流式返回完整消息
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "写一首关于AI的诗"}]},
    stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
```

---

### 2.5 Agent的类型与决策模式

LangChain 内置了多种 Agent 类型，它们在决策方式和适用场景上有所不同。

| Agent 类型 | 核心特点 | 适用场景 |
| :--- | :--- | :--- |
| **ReAct (Reasoning + Acting)** | 强制模型交替输出“思考”和“行动”，决策过程透明。 | 需要清晰展示推理过程的任务，是通用 Agent 的基础。 |
| **Zero-Shot ReAct** | 仅基于工具的描述来决定如何使用工具，无需示例。 | 工具数量不多、功能描述清晰的简单任务。 |
| **Structured Chat ReAct** | 支持向工具传递多个结构化参数（如 JSON）。 | 需要调用复杂 API 或函数，且参数较多的场景。 |
| **Conversational ReAct** | 在 ReAct 基础上增加了**对话记忆**能力。 | 需要多轮对话和上下文理解的聊天机器人。 |
| **Self-Ask with Search** | 将复杂问题拆解成一系列更简单的子问题。 | 需要多步推理和事实核查的问答系统。 |

---

### 2.6 多智能体系统 (Multi-Agent Systems)

当任务过于复杂或涉及多个专业领域时，可以使用多智能体模式。

*   **子智能体模式 (Subagents)**：创建一个**主管 Agent (Supervisor)**，它将多个专注于特定任务的**子 Agent** 封装成工具。主管 Agent 负责决策调用哪个子 Agent。
*   **路由器模式 (Router)**：由一个分类器（Router）根据用户输入，将任务分发给不同的专业 Agent。

**子智能体模式代码示例：**

```python
from langchain.tools import tool
from langchain.agents import create_agent

# 1. 创建一个专门做研究的子 Agent
research_agent = create_agent(model="openai:gpt-4o", tools=[...])

# 2. 将子 Agent 封装成一个工具
@tool("research", description="深入研究一个主题并返回发现")
def call_research_agent(query: str):
    result = research_agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# 3. 创建主管 Agent，并将子 Agent 作为其工具之一
main_agent = create_agent(
    model="openai:gpt-4o",
    tools=[call_research_agent, send_email_tool]
)
```

### 2.7 记忆 (Memory) 的深度集成

LangChain 提供了多种 Memory 实现，以满足不同的场景需求。

*   **`ConversationBufferMemory`**: 简单缓存所有对话。
*   **`ConversationBufferWindowMemory`**: 只保留最近 K 轮对话。
*   **`ConversationSummaryMemory`**: 对历史对话进行摘要压缩。
*   **`VectorStoreRetrieverMemory`**: 使用向量数据库检索相关记忆。


### 2.8 总结

LangChain Agents 是一个功能强大且不断演进的框架。

*   **核心**：建立在“思考-行动-观察”循环之上的**模型-工具-循环**架构。
*   **创建**：**`create_agent`** 是官方推荐的、现代且灵活的 Agent 构建方式。
*   **扩展**：通过 **中间件 (Middleware)** 和**多智能体 (Multi-Agent)** 模式，可以实现极其复杂和定制化的行为。
*   **生产**：关注**工具设计、安全控制、可观测性**和**鲁棒性**是构建可靠应用的关键。

## 三、Middleware（中间件）
### 3.1 介绍
Middleware（中间件）是 LangChain 1.0 中 `create_agent` 的核心特性。你可以把它看作 Agent 生命周期中的“插件”，能在不修改核心逻辑的前提下，以可组合、可复用的方式拦截并增强 Agent 的行为。

Middleware 主要用于解决以下**横切关注点**（Cross-Cutting Concerns）：

*   **可观测性**：添加日志、监控和调试信息。
*   **提示词与上下文工程**：动态修改系统提示词、转换工具选择逻辑或格式化输出。
*   **可靠性与容错**：为模型调用或工具执行添加**重试**（Retries）、**降级**（Fallbacks）和**限流**（Rate Limiting）逻辑。
*   **安全与合规**：实现**PII（个人身份信息）检测与脱敏**、内容审核等安全护栏（Guardrails）。
*   **流程控制**：实现**人机协同**（Human-in-the-Loop）、任务规划等高级控制流。
### 3.2 核心钩子
**“节点式”钩子 (Node-style Hooks)**
这些钩子在特定节点按顺序执行，适用于日志记录、状态验证等场景。

| 钩子 (Hook) | 执行时机 | 典型用途 |
| :--- | :--- | :--- |
| **`before_agent`** | Agent 调用开始前，**只运行一次**。 | 加载记忆、初始化资源、验证输入。 |
| **`before_model`** | **每次**模型调用前。 | 动态修改系统提示词、注入上下文、进行 PII 检测。 |
| **`after_model`** | **每次**模型响应后，但在执行工具前。 | 检查模型输出、实现人机协同审批。 |
| **`after_agent`** | Agent 调用完成后，**只运行一次**。 | 保存最终状态、发送通知、清理资源。 |

**“包装式”钩子 (Wrap-style Hooks)**
这些钩子“包裹”住核心调用，让你能完全控制其执行过程，适用于重试、缓存等场景。

| 钩子 (Hook) | 执行时机 | 典型用途 |
| :--- | :--- | :--- |
| **`wrap_model_call`** | 包裹**每次**模型调用。 | 实现模型调用的重试、缓存、动态更换模型。 |
| **`wrap_tool_call`** | 包裹**每次**工具执行。 | 实现工具调用的重试、缓存、权限检查、结果后处理。 |

### 3.3 内置中间件 (Built-in Middleware)

LangChain 提供了丰富的预置中间件，覆盖了最常见的需求。以下是一些核心内置中间件：

*   **`SummarizationMiddleware`**：在对话历史接近 Token 限制时自动进行摘要，防止超出上下文窗口。
*   **`HumanInTheLoopMiddleware`**：对敏感工具调用（如发送邮件、删除数据）暂停执行，等待人类审批。
*   **`PIIMiddleware`**：自动检测并脱敏输入、输出中的个人身份信息（如姓名、邮箱、电话号码）。
*   **`ModelFallbackMiddleware`**：当主模型调用失败时，自动切换到备用模型。
*   **`ToolRetryMiddleware`**：当工具执行失败时，自动进行重试（支持指数退避）。
*   **`LLMToolSelectorMiddleware`**：当可用工具非常多时，先用一个 LLM 筛选出与当前问题最相关的几个工具。
*   **`TodoListMiddleware`**：为 Agent 配备任务规划与追踪能力。
*   **`ModelCallLimitMiddleware`**：限制模型调用次数，防止 Agent 陷入死循环或产生意外的高额费用。

### 3.4 使用示例

**1. 使用内置中间件**
使用内置中间件非常简单，只需在创建 Agent 时通过 `middleware` 参数传入即可。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware
)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        # 当历史消息超过4000 tokens时触发摘要，并保留最近20条消息
        SummarizationMiddleware(
            model="gpt-3.5-turbo",  # 用于生成摘要的模型
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
        # 对 "send_email" 这个工具调用需要人工审批
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
        # 自动进行PII脱敏
        PIIMiddleware(),
    ]
)
```

**2. 创建自定义中间件**

**方式一：使用装饰器 (Decorator)**
这是最简洁的方式，适合创建功能单一的中间件。

```python
from langchain.agents.middleware import before_model, AgentState
from langgraph.runtime import Runtime
from typing import Any

# 在每次模型调用前，动态注入当前时间
@before_model
def inject_current_time(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 在系统提示词末尾追加当前时间信息
    return {
        "system_prompt": f"Current time is: {current_time}\n"
    }

# 将这个函数作为中间件传入
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_current_time]
)
```

**方式二：继承 `AgentMiddleware` 基类**
这种方式更结构化，适合需要管理状态或组合多个钩子的复杂中间件。

```python
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from typing import Any

class LoggingMiddleware(AgentMiddleware):
    "一个用于记录日志的中间件"

    async def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[LOG] Agent invocation started.")
        return None

    async def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        print(f"[LOG] Model responded with: {last_message.content[:50]}...")
        return None

    async def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[LOG] Agent invocation finished.")
        return None

# 使用自定义中间件
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[LoggingMiddleware()]
)
```

**高级用法与最佳实践**

*   **状态更新 (State Updates)**：节点式钩子（如 `before_model`）可通过返回 `dict` 来更新 Agent 状态。包装式钩子（如 `wrap_model_call`）则需返回 `ExtendedModelResponse` 或 `Command` 对象来注入状态更新。

*   **组合与顺序 (Composition & Order)**：多个中间件可以像“洋葱”一样组合。节点式钩子按列表顺序执行，而包装式钩子（`wrap_model_call`）则是**反向**执行的。

*   **跳转控制 (Jump To)**：节点式钩子可以通过返回 `{"jump_to": "end"}` 来提前终止 Agent 的执行。

*   **设计原则**：
    *   **单一职责**：每个中间件只做好一件事。
    *   **优雅处理错误**：捕获并妥善处理中间件内部的异常，避免导致整个 Agent 崩溃。
    *   **选择合适的钩子**：根据你的需求，选择最合适的钩子类型。

## 三、Retrievers检索器
### 3.1 介绍

Retriever（检索器）是 LangChain 中的一个核心接口，它接受一个非结构化的字符串查询，并返回最相关的 Document 对象列表。检索器比向量存储（Vector Store）更通用——它不需要能够存储文档，只需要能够返回（检索）文档即可。所有向量存储都可以通过 `as_retriever()` 方法转换为检索器。

检索器遵循 LangChain 的 `Runnable` 接口，支持以下标准方法：
- `invoke(query)` — 同步调用
- `ainvoke(query)` — 异步调用
- `batch([queries])` — 批量调用
- `abatch([queries])` — 异步批量调用

**常用方法总结**

所有检索器都继承自 `BaseRetriever`，支持以下方法：
| 方法 | 说明 |
|------|------|
| `invoke(query)` | 同步检索，输入字符串查询，返回 Document 列表 |
| `ainvoke(query)` | 异步检索 |
| `batch([queries])` | 批量同步检索，输入查询列表，返回 Document 列表的列表 |
| `abatch([queries])` | 批量异步检索 |
| `get_relevant_documents(query)` | 旧版同步检索方法（建议使用 `invoke`） |
| `aget_relevant_documents(query)` | 旧版异步检索方法（建议使用 `ainvoke`） |
**批量操作示例**：
```python
queries = [
    "What is Mount Everest?",
    "Tell me about the Eiffel Tower.",
    "What is the Great Wall of China?"
]
batch_results = retriever.batch(queries)
for i, results in enumerate(batch_results):
    print(f"Results for query: {queries[i]}")
    for doc in results:
        print(doc.page_content)
```


**常用检索器：**
VectorStoreRetriever（向量存储检索器）
MultiQueryRetriever（多查询检索器）
EnsembleRetriever（集成检索器）
ContextualCompressionRetriever（上下文压缩检索器）
ParentDocumentRetriever（父文档检索器）
SelfQueryRetriever（自查询检索器）
TimeWeightedVectorStoreRetriever（时间加权检索器）


### 2. VectorStoreRetriever（向量存储检索器）

最基础的检索器，从向量存储中基于相似度检索文档。

**创建方式**：

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 创建向量存储
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# 转换为检索器
retriever = vectorstore.as_retriever()
```

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `search_type` | str | 搜索类型：`"similarity"`（相似度）、`"mmr"`（最大边际相关性）、`"similarity_score_threshold"`（相似度阈值） |
| `search_kwargs` | dict | 搜索参数，如 `k`（返回文档数量） |

**完整示例**：

```python
# 相似度搜索，返回前2个文档
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# MMR搜索，平衡相关性与多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# 相似度阈值搜索
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 10}
)

# 执行检索
results = retriever.invoke("Where is the Eiffel Tower located?")
for doc in results:
    print(doc.page_content)
```


### 3. MultiQueryRetriever（多查询检索器）

使用 LLM 从不同角度生成多个查询变体，分别检索后合并去重，以提高召回率。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `retriever` | BaseRetriever | 基础检索器 |
| `llm` | BaseLLM | 用于生成查询的 LLM |
| `query_prompt` | PromptTemplate | 自定义提示模板（可选） |
| `verbose` | bool | 是否输出详细日志 |

**完整示例**：

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=ChatOpenAI(model="gpt-4"),
    verbose=True
)

docs = retriever.invoke("How many people were injured in the earthquake?")
```


### 4. EnsembleRetriever（集成检索器）

使用排名融合（Rank Fusion）算法集成多个检索器的结果。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `retrievers` | List[BaseRetriever] | 要集成的检索器列表 |
| `weights` | List[float] | 各检索器的权重，默认等权重 |
| `c` | int | 排名融合常数，默认60，控制高排名与低排名项目的平衡 |
| `id_key` | str | 用于去重的文档元数据键名，默认使用 `page_content` |

**完整示例**：

```python
from langchain.retrievers import EnsembleRetriever

# 假设有两个不同的检索器
retriever1 = vectorstore1.as_retriever(search_kwargs={"k": 5})
retriever2 = vectorstore2.as_retriever(search_kwargs={"k": 5})

ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2],
    weights=[0.6, 0.4],  # 权重分配
    c=60
)

docs = ensemble_retriever.invoke("What is artificial intelligence?")
```


### 5. ContextualCompressionRetriever（上下文压缩检索器）

包装基础检索器，对检索结果进行压缩和过滤，只返回最相关的信息。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `base_compressor` | BaseDocumentCompressor | 文档压缩器 |
| `base_retriever` | BaseRetriever | 基础检索器 |

**完整示例**：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4"))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})
)

# 压缩后的文档只包含最相关的信息
docs = compression_retriever.invoke("What is the capital of France?")
```


### 6. ParentDocumentRetriever（父文档检索器）

检索小文本块，然后返回其对应的父文档（更大的文本块），在细粒度检索和充足上下文之间取得平衡。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `vectorstore` | VectorStore | 存储小文本块的向量存储 |
| `docstore` | BaseStore | 存储父文档的文档存储 |
| `child_splitter` | TextSplitter | 子文档分割器 |
| `parent_splitter` | TextSplitter | 父文档分割器（可选） |
| `id_key` | str | 文档ID的元数据键名 |

**完整示例**：

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# 添加文档
retriever.add_documents(documents)

# 检索时返回父文档
docs = retriever.invoke("What is machine learning?")
```


### 7. SelfQueryRetriever（自查询检索器）

使用 LLM 生成结构化查询，将自然语言查询转换为向量存储的过滤条件。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `llm` | BaseLLM | 用于生成查询的 LLM |
| `vectorstore` | VectorStore | 向量存储 |
| `document_contents` | str | 文档内容描述 |
| `metadata_field_info` | List | 元数据字段信息 |
| `structured_query_translator` | Translator | 查询语言转换器（可选） |
| `enable_limit` | bool | 是否启用结果数量限制 |
| `verbose` | bool | 是否输出详细日志 |

**完整示例**：

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    # 定义元数据字段，如 "year", "author", "category" 等
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    vectorstore=vectorstore,
    document_contents="Brief summary of a document",
    metadata_field_info=metadata_field_info,
    verbose=True,
    enable_limit=True
)

# 支持带过滤条件的自然语言查询
docs = retriever.invoke("Show me documents about AI from 2023")
```


### 8. TimeWeightedVectorStoreRetriever（时间加权检索器）

结合嵌入相似度和文档新鲜度（时效性）进行检索。

**参数详解**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `vectorstore` | VectorStore | 向量存储 |
| `decay_rate` | float | 衰减率，控制时效性权重 |
| `k` | int | 返回文档数量 |
| `other_scores` | dict | 其他评分参数 |

**评分公式**：`semantic_similarity + (1.0 - decay_rate) ** hours_passed`

**完整示例**：

```python
from langchain.retrievers import TimeWeightedVectorStoreRetriever

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.0000000000000000000000001,
    k=5
)

# 注意：必须通过检索器的 add_documents 方法添加文档，而非直接操作 vectorstore
retriever.add_documents(documents)

docs = retriever.invoke("Recent developments in quantum computing")
```


### 9. 其他社区检索器（langchain_community）

LangChain 社区提供了大量第三方检索器集成：

| 检索器 | 说明 |
|--------|------|
| `ArxivRetriever` | arXiv 学术论文检索 |
| `WikipediaRetriever` | Wikipedia 检索 |
| `PubMedRetriever` | PubMed 医学文献检索 |
| `BM25Retriever` | BM25 算法检索（无需 Elasticsearch） |
| `ElasticSearchBM25Retriever` | Elasticsearch BM25 检索 |
| `SVMRetriever` | 基于 SVM 的检索 |
| `TFIDFRetriever` | TF-IDF 检索 |
| `KNNRetriever` | KNN 检索 |
| `AzureAISearchRetriever` | Azure AI 搜索检索 |
| `MilvusRetriever` | Milvus 向量数据库检索 |
| `TavilySearchAPIRetriever` | Tavily 搜索 API |
| `ChaindeskRetriever` | Chaindesk API 检索 |
| `MetalRetriever` | Metal API 检索 |
| `ZepRetriever` | Zep MemoryStore 检索 |
| `VespaRetriever` | Vespa 检索 |
| `LlamaIndexRetriever` | LlamaIndex 检索器适配 |
| `ChatGPTPluginRetriever` | ChatGPT 插件检索 |
| `RemoteLangChainRetriever` | 远程 LangChain API 检索 |

**使用示例**：

```python
from langchain_community.retrievers import ArxivRetriever, WikipediaRetriever

# arXiv 检索
arxiv_retriever = ArxivRetriever(
    load_max_docs=5,
    get_full_documents=True
)
docs = arxiv_retriever.invoke("quantum machine learning")

# Wikipedia 检索
wiki_retriever = WikipediaRetriever(
    lang="en",
    load_max_docs=3
)
docs = wiki_retriever.invoke("Artificial intelligence")
```

### 10、完整示例

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever

# 1. 加载和分割文档
loader = TextLoader("documents.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 2. 创建向量存储
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# 3. 创建基础检索器
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# 4. 创建多查询检索器（提高召回率）
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(model="gpt-4")
)

# 5. 创建上下文压缩检索器（提高精度）
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4"))
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_retriever
)

# 6. 执行检索
query = "What are the key findings about climate change?"
docs = compression_retriever.invoke(query)

for doc in docs:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Metadata: {doc.metadata}")
```




























