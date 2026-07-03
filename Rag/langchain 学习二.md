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

## 1.4 Toolkits工具包
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







