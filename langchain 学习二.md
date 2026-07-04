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

























