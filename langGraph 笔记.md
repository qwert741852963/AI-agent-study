
## 一、三大核心组件
### 1.1 state状态

#### 1.1.1 定义状态（State）

LangGraph 支持多种方式来定义状态的结构。

**1. 使用 TypedDict（Python 推荐）**

这是最常用的方法，简单直观。

```python
from typing import List
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage

class State(TypedDict):
    messages: List[AnyMessage]  # 消息列表
    extra_field: int            # 额外整数字段
```

**2. 使用 Pydantic BaseModel**
提供数据验证功能。

```python
from pydantic import BaseModel
from langchain_core.messages import AnyMessage

class State(BaseModel):
    messages: List[AnyMessage]
    extra_field: int
```

**3. 使用 `Annotation.Root`（TypeScript / JavaScript）**

在 JS/TS 中，使用 `Annotation.Root` 来定义状态，并可以内联指定 Reducer。

```typescript
import { Annotation } from "@langchain/langgraph";

// 1. 简单状态：无 reducer，默认覆盖
const SimpleState = Annotation.Root({
    currentOutput: Annotation<string>,
});

// 2. 带 reducer 的状态
const MessagesState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
        reducer: (left, right) => left.concat(right), // 合并数组
        default: () => [],
    }),
});
```


#### 1.1.2 Reducer（归约器）
##### Reducer 的用法
**Reducer（归约器）是 LangGraph 中控制状态如何更新的核心函数**。简单来说，当一个节点（Node）执行完成后返回更新数据时，Reducer 决定了这份新数据**如何与状态（State）中的现有数据融合**。

其数学表达为：

```
new_state = reducer(current_state, update_value)
```

Reducer 接收两个参数：
- **当前状态值**（current state）
- **节点返回的更新值**（update value）

然后返回合并后的新状态。

**为什么需要 Reducer？** 默认情况下，LangGraph 对状态字段采用**直接覆盖**策略。但在实际场景中，我们往往需要更复杂的更新逻辑——比如将新消息追加到对话历史末尾，而不是覆盖掉之前的消息。Reducer 正是为此而生。

**1. 使用 `Annotated` 类型定义 Reducer（Python）**
在 Python 中，通过 `typing.Annotated` 将 Reducer 函数与状态字段绑定：

```python
from typing import Annotated, TypedDict
from operator import add

class State(TypedDict):
    # 使用 operator.add 作为 reducer：每次更新时做累加
    counter: Annotated[int, add]
    
    # 没有标注 reducer 的字段，默认使用覆盖策略
    name: str
```

---

##### Reducer 类型与示例

**1. 默认 Reducer：覆盖更新**

当状态字段**没有**指定 Reducer 时，默认行为是**直接覆盖**。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    counter: int

def node_a(state: State):
    return {"counter": state["counter"] + 1}

def node_b(state: State):
    return {"counter": state["counter"] * 2}

builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()
result = graph.invoke({"counter": 5})
print(result)  # 输出: {'counter': 12}
# node_a 返回 counter=6，node_b 返回 counter=12，后者覆盖前者
```

**2. `operator.add`：数值累加**

适用于计数器、分数聚合、资源统计等场景。

```python
from typing import Annotated, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    total: Annotated[int, add]  # 累加器

def node_a(state: State):
    return {"total": 10}

def node_b(state: State):
    return {"total": 5}

builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

graph = builder.compile()
result = graph.invoke({"total": 0})
print(result)  # 输出: {'total': 15}  (0 + 10 + 5)
```

**3. `add_messages`：消息列表追加**

这是 LangGraph 内置的**消息专用 Reducer**，专门用于对话系统。它会将新消息追加到消息列表末尾，并支持消息的去重、替换和删除。

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AIMessage, HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    # 模拟 LLM 回复
    return {"messages": [AIMessage(content="你好！有什么我可以帮你的？")]}

builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
result = graph.invoke({
    "messages": [HumanMessage(content="你好")]
})
print(result["messages"])
# 输出: [HumanMessage(content="你好"), AIMessage(content="你好！有什么我可以帮你的？")]
# 新消息被追加，而不是覆盖
```

**4. 自定义 Reducer**

当内置 Reducer 无法满足需求时，可以编写自定义逻辑。

```python
from typing import Annotated, TypedDict

def merge_list_reducer(current: list, update: list | None) -> list:
    """合并两个列表，去重并保持顺序"""
    if update is None:
        return current
    # 合并并去重（保持原有顺序）
    seen = set(current)
    result = current.copy()
    for item in update:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

class State(TypedDict):
    items: Annotated[list, merge_list_reducer]

# 使用方式与前述相同
```

---

**为什么 Reducer 在并行执行中至关重要？**

当图中有**多个节点并行执行**（如 Fan-out）并同时更新**同一个状态字段**时，如果没有 Reducer，LangGraph 无法确定如何合并这些并发更新，会抛出 `INVALID_CONCURRENT_GRAPH_UPDATE` 错误。
**Reducer 解决了这个问题**——它定义了明确的合并逻辑，让 LangGraph 知道如何将多个并行节点的输出安全地合并到同一状态字段中。

---

**总结**

| Reducer 类型 | 行为 | 适用场景 |
|---|---|---|
| **默认（无 Reducer）** | 直接覆盖 | 独立计算结果、配置更新 |
| **`operator.add`** | 数值累加 | 计数器、分数聚合 |
| **`add_messages`** | 消息追加 | 对话历史、聊天系统 |
| **自定义 Reducer** | 自定义逻辑 | 复杂合并需求、去重、条件更新 |

**核心要点**：
1. Reducer 决定了状态**如何更新**，而非**是否更新**
2. 通过 `Annotated`（Python）或 `Annotation`（JS）将 Reducer 与字段绑定
3. 并行执行时必须为共享字段定义 Reducer，避免并发冲突

#### 1.1.3 状态相关方法与参数

| 方法/属性 | 描述 | 示例 |
| :--- | :--- | :--- |
| **`StateGraph(state_schema)`** | 构建图时传入状态定义（类或 `Annotation.Root`）。 | `builder = StateGraph(State)` |
| **`.compile()`** | 编译图，使其可执行。可传入 `checkpointer` 等参数以启用持久化。 | `graph = builder.compile(checkpointer=memory)` |
| **`.invoke(input_state)`** | 同步执行图，输入初始状态，返回最终状态。 | `result = graph.invoke({"messages": []})` |
| **`.stream(input_state)`** | 流式执行，可以逐步骤获取状态更新（`updates`）或完整状态（`values`）。 | `for update in graph.stream(...): print(update)` |
| **`.get_state(config)`** | 获取某个配置（如线程ID）下的当前状态，用于持久化和检查点。 | `current = graph.get_state({"configurable": {"thread_id": "1"}})` |
| **`.update_state(config, values)`** | 手动更新指定配置下的状态。 | `graph.update_state(config, {"messages": [...]})` |

### 1.2 Node节点

#### 1.2.1 节点的定义方式

在 LangGraph 中，**节点（Node）** 是图的基本执行单元，本质上是**接收当前状态（State）并返回更新后状态的 Python/TypeScript 函数**。每个节点负责完成一个特定的任务，例如调用 LLM、执行工具调用或进行条件判断。

> 核心思想：**节点完成工作，边（Edge）指示下一步该做什么**。


最简单的节点就是一个普通函数：

```python
from typing_extensions import TypedDict

class State(TypedDict):
    x: int

def my_node(state: State) -> dict:
    # 读取状态，执行逻辑，返回更新
    return {"x": state["x"] + 1}
```

节点的**第一个参数始终是当前的 State**，返回值是部分状态更新（`Partial<State>`）。

如果需要访问运行时上下文（如 `user_id`、数据库连接等），可以添加第二个参数：

```python
from langchain_core.runnables import RunnableConfig

class Context(TypedDict):
    user_id: str

def node_with_context(state: State, config: RunnableConfig) -> dict:
    user_id = config.configurable.get("user_id")
    # 执行业务逻辑...
    return {"x": state["x"] + 1}
```



#### 1.2.2 `add_node` 方法详解

将节点添加到图中使用 `add_node` 方法：

**Python 语法**

```python
builder.add_node(
    node,           # 节点名称（str）或节点函数本身
    action,         # 当 node 为字符串时，指定实际的节点函数
    *,
    defer=False,    # 是否延迟执行
    metadata=None,  # 元数据信息
    retry_policy=None,  # 重试策略
    cache_policy=None,   # 缓存策略
    timeout=None,        # 超时设置
)
```

**关键参数说明**

| 参数 | 类型 | 说明 |
|------|------|------|
| `node` | `str` 或 `StateNode` | 节点名称或节点函数 |
| `action` | `StateNode` | 当 `node` 为字符串时，指定实际执行的函数 |
| `defer` | `bool` | 是否延迟执行该节点 |
| `retry_policy` | `RetryPolicy` | 节点失败时的重试策略 |
| `timeout` | `float` / `timedelta` | 节点执行超时时间 |

**使用示例**

```python
from langgraph.graph import StateGraph, START

builder = StateGraph(State)

# 方式一：直接传入函数（自动使用函数名作为节点名）
builder.add_node(my_node)

# 方式二：自定义节点名称
builder.add_node("my_fair_node", my_node)

# 方式三：使用匿名函数
builder.add_node("lambda_node", lambda state: {"x": state["x"] * 2})

# 连接节点
builder.add_edge(START, "my_fair_node")
graph = builder.compile()

result = graph.invoke({"x": 1})  # {'x': 2}
```

---

#### 1.2.3 节点的高级用法

**1. 使用 `Command` 进行路由控制**
节点可以返回 `Command` 对象，同时完成状态更新和路由跳转：

```python
from langgraph.graph import Command

def router_node(state: State) -> Command:
    if state.get("needs_tool"):
        return Command(
            goto="tool_node",           # 跳转到工具节点
            update={"step": state["step"] + 1}
        )
    return Command(goto="agent_node")   # 跳转到智能体节点
```

**2. 使用 `ToolNode` 预置节点**

LangGraph 提供了预置的 `ToolNode`，用于执行工具调用：

```python
from langgraph.prebuilt import ToolNode

# 定义工具列表
tools = [search_tool, calculator_tool]
tool_node = ToolNode(tools)

builder.add_node("tools", tool_node)
```


#### 1.2.4 完整示例
**构建一个简单聊天机器人**
```python
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage

# 1. 定义状态
class State(TypedDict):
    messages: list
    step: int

# 2. 定义节点函数
def chat_node(state: State) -> dict:
    # 模拟 LLM 响应
    last_msg = state["messages"][-1]["content"] if state["messages"] else ""
    response = f"你说的是：{last_msg}，我收到了！"
    return {
        "messages": [{"role": "assistant", "content": response}],
        "step": state.get("step", 0) + 1
    }

def check_node(state: State) -> dict:
    # 检查是否应该结束
    if state.get("step", 0) >= 3:
        return {"messages": [{"role": "system", "content": "对话结束"}]}
    return {}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_node("check", check_node)

builder.add_edge(START, "chat")
builder.add_edge("chat", "check")
builder.add_edge("check", END)

# 4. 编译并执行
graph = builder.compile()
result = graph.invoke({
    "messages": [{"role": "user", "content": "你好！"}],
    "step": 0
})
print(result)
```

### 1.3 边（Edges）

在 LangGraph 中，**边（Edges）** 是连接**节点（Nodes）**、定义程序执行流向的核心组件。简单来说，`Nodes` 负责“做什么”，而 `Edges` 负责“接下来做什么”。

#### 1.3.1 核心方法

在 `StateGraph` 构建器中，主要通过两个方法来定义边：

*   **`add_edge(source, target)`**: 定义一条**普通边（Normal Edge）**，表示从 `source` 节点执行完后，**无条件地**、**总是**去执行 `target` 节点。
*   **`add_conditional_edges(source, path, path_map=None)`**: 定义一条**条件边（Conditional Edge）**，它会在 `source` 节点执行完后，**根据当前状态（State）** 动态决定下一个要执行的节点。

LangGraph 预置了两个特殊节点常量：
*   `START`: 图的入口。
*   `END`: 图的终点，执行到此处即终止。

**1. `add_edge`**

此方法用于建立确定性的执行路径。

*   **参数**:
    *   `source` (str): 起始节点的名称。
    *   `target` (str): 目标节点的名称。
*   **行为**:
    *   从 `START` 连接到第一个节点，定义入口。
    *   连接到 `END` 表示流程结束。
    *   如果一个节点有多个出边（例如 `add_edge("A", "B")` 和 `add_edge("A", "C")`），则 `B` 和 `C` 会在同一个**超级步（Superstep）** 中**并行执行**。

**2. `add_conditional_edges`**

此方法用于实现动态路由。

*   **参数**:
    *   `source` (str): 起始节点的名称。
    *   `path` (Callable): **路由函数（Routing Function）**。它接收当前 `State` 作为参数，并返回一个字符串（或字符串列表），用于指示下一个节点的名称。
    *   `path_map` (dict, optional): **路径映射**。一个可选字典，用于将 `path` 函数的返回值映射到实际的节点名称。这在返回值是符号（如 "continue", "exit"）而非确切节点名时很有用。
*   **行为**:
    *   `path` 函数可以返回单个节点名（分支）或一个节点名列表（扇出，Fan-out）。
    *   若提供了 `path_map`，LangGraph 会根据映射关系找到目标节点。

#### 1.3.2 代码示例

**示例 1：普通边 (固定流转)**
这个例子展示了如何使用 `add_edge` 构建一个简单的顺序工作流。

```python
from langgraph.graph import StateGraph, END

# 定义节点函数
def step_a(state):
    state["result"] = "A完成"
    return state

def step_b(state):
    state["result"] += " -> B完成"
    return state

def step_c(state):
    state["result"] += " -> C完成"
    return state

# 1. 构建图
builder = StateGraph(dict)

# 2. 添加节点
builder.add_node("step_a", step_a)
builder.add_node("step_b", step_b)
builder.add_node("step_c", step_c)

# 3. 添加边 (固定流转)
builder.add_edge(START, "step_a")  # 从 START 开始
builder.add_edge("step_a", "step_b") # A 完成后执行 B
builder.add_edge("step_b", "step_c") # B 完成后执行 C
builder.add_edge("step_c", END)      # C 完成后结束

# 4. 编译并运行
graph = builder.compile()
result = graph.invoke({"result": ""})
print(result) # 输出: {'result': 'A完成 -> B完成 -> C完成'}
```

**示例 2：条件边 (动态路由)**

这个例子展示了如何根据用户输入，使用 `add_conditional_edges` 动态路由到不同节点。

```python
from langgraph.graph import StateGraph, END
from typing import Literal

# 1. 定义 State
class State(dict):
    query: str
    response: str

# 2. 定义节点函数
def classifier(state: State):
    return state # 此节点仅用于路由判断，直接返回 state

def price_handler(state: State):
    state["response"] = "正在查询价格..."
    return state

def after_sales_handler(state: State):
    state["response"] = "转接售后专员..."
    return state

def general_handler(state: State):
    state["response"] = "这是通用回答..."
    return state

# 3. 定义路由函数
def route_query(state: State) -> Literal["price", "after_sales", "general"]:
    query = state.get("query", "")
    if "价格" in query or "多少钱" in query:
        return "price"
    elif "售后" in query or "维修" in query:
        return "after_sales"
    else:
        return "general"

# 4. 构建图
builder = StateGraph(State)
builder.add_node("classifier", classifier)
builder.add_node("price", price_handler)
builder.add_node("after_sales", after_sales_handler)
builder.add_node("general", general_handler)

# 5. 设置入口并添加条件边
builder.set_entry_point("classifier") # 从 classifier 开始
builder.add_conditional_edges(
    "classifier",            # 起始节点
    route_query,             # 路由函数
    {                        # 路径映射 (可选)
        "price": "price",
        "after_sales": "after_sales",
        "general": "general"
    }
)
# 为三个业务节点添加指向 END 的固定边
builder.add_edge("price", END)
builder.add_edge("after_sales", END)
builder.add_edge("general", END)

# 6. 编译并运行
graph = builder.compile()
result = graph.invoke({"query": "我想问一下价格"})
print(result) # 输出: {'query': '我想问一下价格', 'response': '正在查询价格...'}
```

**示例 3：条件边 + `path_map` (映射路由)**

`path_map` 在你希望路由函数的返回值更具描述性，而非直接等于节点名时非常有用。

```python
from langgraph.graph import StateGraph, END
import random

# 定义节点 (node_1, node_2, node_3 略, 与示例1类似)
# ...

# 定义路由函数，返回描述性字符串
def decide_mood(state) -> str:
    if random.random() < 0.5:
        return "happy"  # 返回描述性标签
    else:
        return "sad"

builder = StateGraph(State)
# ... 添加 node_1, node_2, node_3 ...

# 使用 path_map 将描述性标签映射到实际节点名
builder.add_conditional_edges(
    "node_1",             # 起始节点
    decide_mood,          # 路由函数
    {                     # path_map
        "happy": "node_2",
        "sad": "node_3"
    }
)
```

#### 1.3.3 高级路由

除了基本的条件边，LangGraph 还提供了更高级的控制流原语：

*   **`Command`**: 允许节点在返回时**同时更新状态（State）并指定下一个节点**，逻辑更集中。
*   **`Send`**: 用于**动态扇出（Dynamic Fan-out）**。条件边的路由函数可以返回一个 `Send` 对象列表，每个 `Send` 对象能携带不同的状态数据给不同的节点，实现 Map-Reduce 模式。
*   **`interrupt()`**: 用于实现**人机协作（Human-in-the-loop）**，在节点执行过程中暂停图，等待外部输入后再继续。

**注意事项**
*   **编译**：在添加完所有节点和边后，必须调用 `.compile()` 方法对图进行编译，才能执行。
*   **状态更新**：节点函数需要返回一个**状态更新（State Update）**，LangGraph 会使用 `reducer` 函数（如果定义了）将此更新合并到全局状态中。
*   **并行执行**：当一个节点通过多条普通边连接到多个后续节点时，这些后续节点会**并行执行**。

简单来说，**普通边 (`add_edge`) 用于确定的、无分支的流程**，而 **条件边 (`add_conditional_edges`) 用于根据当前状态做出决策的动态流程**。理解并灵活运用这两种边，是构建复杂 LangGraph 应用的基础。


## 二、边的高级用法（Command和Send）

### 2.1 介绍

`Command` 和 `Send` 是 LangGraph 中用于实现**动态控制流**的两种高级原语，它们解决了传统“节点-边”静态结构无法灵活应对的复杂场景。

简单来说，`Command` 侧重于 **“节点内部自主决策下一站”** ，而 `Send` 侧重于 **“动态创建并行任务”** 。下表可以帮你快速理解它们的核心区别：

| 特性 | `Command` | `Send` |
| :--- | :--- | :--- |
| **核心作用** | 节点在执行完毕后，**自主决定**下一步去哪个节点。 | 在条件边中，**动态地、并行地**将任务派发给一个或多个节点。 |
| **使用场景** | 多智能体交接、节点内动态路由、替代简单的条件边。 | Map-Reduce 模式、批量数据处理、需要并行执行未知数量任务的场景。 |
| **返回位置** | 在**节点函数内部**直接返回。 | 在**条件边的路由函数**中返回一个列表。 |
| **关键参数** | `update` (状态更新), `goto` (目标节点)。 | `node` (目标节点名), `arg` (传递给该节点的状态)。 |

---

### 2.2 Command

`Command` 允许一个节点在返回时，**同时完成两件事**：更新状态 (`update`) 和指定下一个要执行的节点 (`goto`)。这使得流程控制逻辑从“边”转移到了“节点”内部，更加灵活和直观。

**核心参数**
*   **`update`** (`dict`): 要更新的状态数据，和普通节点返回的更新一样。
*   **`goto`** (`str` 或 `List[str]`): 指定下一个要执行的节点名称。
*   **`graph`** (`Command.PARENT`): 在子图中使用时，可通过此参数跳转到父图中的节点。
*   **`resume`**: 与 `interrupt()` 配合使用，用于恢复被暂停的图。

**基本用法示例**
假设我们有一个节点，需要根据内部逻辑决定去往 `node_b` 还是 `node_c`。

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from typing import Literal

# 定义状态
class State(dict):
    foo: str

# 定义节点，返回类型注解指明了可能跳转的目标节点
def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    # 模拟一个动态决策
    import random
    goto = "node_b" if random.random() > 0.5 else "node_c"
    
    # 同时更新状态并指定下一站
    return Command(
        update={"foo": "a"},  # 状态更新
        goto=goto             # 流程控制
    )

def node_b(state: State):
    print("执行 B")
    return {"foo": state.get("foo", "") + "|b"}

def node_c(state: State):
    print("执行 C")
    return {"foo": state.get("foo", "") + "|c"}

# 构建图
builder = StateGraph(State)
builder.add_node("node_a", node_a, ends=["node_b", "node_c"]) # ends参数帮助可视化
builder.add_node("node_b", node_b)
builder.add_node("node_c", node_c)
builder.add_edge("__start__", "node_a")
builder.add_edge("node_b", END)
builder.add_edge("node_c", END)

graph = builder.compile()
# 此时，图中不存在从 node_a 到 node_b 或 node_c 的显式边
# 流程完全由 node_a 返回的 Command 决定
```

**典型应用场景：多智能体交接 (Handoff)**
`Command` 是实现多智能体系统（Multi-agent System）中“交接”的理想工具。一个智能体（节点）在处理完任务后，可以通过 `Command` 直接将控制权“交接”给另一个专门的智能体（节点）。

```python
def agent_router(state):
    # ... 分析用户意图 ...
    if "天气" in state["query"]:
        # 交接给天气智能体，并传递相关信息
        return Command(goto="weather_agent", update={"intent": "weather"})
    else:
        return Command(goto="default_agent")
```

---

### 2.3 Send：动态并行的“任务分派器”

你是一个项目经理，今天要处理一个任务清单：
```
任务清单 = ["写报告", "做PPT", "发邮件"]
```
你不可能自己一个人干完，所以你**动态**地（根据清单的内容）为**每一个**任务都**招聘**一个员工去执行，而且他们可以**同时开工**。

这里的关键点有三个：

1. **任务数量不确定**：今天是3个，明天可能是10个，没法在代码里写死。
2. **每个任务处理的数据不同**：一个员工处理“写报告”，一个处理“做PPT”。
3. **并行执行**：大家同时干，效率最高。

如果用普通的图（边）来做，你只能画死 `A -> B -> C`，无法应对这种“清单长度会变化”的情况。

**`Send` 就是用来做这件事的**：它让你在图的运行过程中，**根据当前状态里的数据，动态地、并行地产生任意数量的新任务**。

---

**🧱 拆解 `Send` 的两个参数**
`Send` 本身非常简单，就两个参数：

| 参数 | 含义 | 生活化类比 |
| :--- | :--- | :--- |
| **`node`** | 你要把任务派给**哪个节点（员工）** 去做。 | “这个任务交给哪位同事？” |
| **`arg`** | 你要传给那个节点的**专属数据**。 | “交给他的具体工作内容和资料是什么？” |

注意：`arg` 里的数据结构，**可以和你主图的状态（State）结构不同**。比如主图状态是 `{"任务清单": [...]}`，但你用 `Send` 传过去的是 `{"单个任务": "写报告"}`，这样每个员工只关心自己的那一小块工作，更清晰。

---

**📖 一个让你彻底明白的例子**

这次我们做一个**“批量文件处理”**的例子，逻辑简单直接。

**场景:**
主图接收到一个文件列表 `["a.txt", "b.txt", "c.txt"]`，我们要为**每一个文件**启动一个独立的 `processor` 节点来处理它。

```python
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing import List, Annotated
import operator

# 1. 定义主图的状态
# files: 待处理的文件列表
# results: 所有处理结果的汇总（使用 operator.add 自动累加）
class OverallState(dict):
    files: List[str]
    results: Annotated[List[str], operator.add]

# 2. 定义“分配任务”的函数（这是一个条件边的路由函数）
def assign_worker(state: OverallState):
    """
    这个函数根据主状态里的 files 列表，动态产生一堆 Send 任务。
    """
    # 关键点：遍历列表，为每一个文件创建一个 Send 对象
    send_list = []
    for file_name in state["files"]:
        # Send(node=目标节点名, arg=传给该节点的状态)
        # 注意：arg 里的结构是 {"file": file_name}，与主状态 OverallState 不同
        send_list.append(
            Send(
                node="processor",          # 所有任务都发给 processor 这个节点
                arg={"file": file_name}    # 但每个任务携带的数据不同
            )
        )
    return send_list  # 返回 Send 对象列表

# 3. 定义“工作节点”的函数
def processor(state: dict):
    """
    这个节点接收来自 Send 的 arg 数据，只处理一个文件。
    state 在这里是 {"file": "a.txt"} 这样的结构。
    """
    file = state["file"]
    # 模拟处理，生成一个结果
    result = f"已处理: {file}"
    # 返回的结果会被累加到主状态的 results 字段中（因为用了 operator.add）
    return {"results": [result]}

# 4. 构建图
builder = StateGraph(OverallState)

# 添加节点
builder.add_node("processor", processor)  # 工作节点

# 关键：从 START 开始，添加一条条件边
# 这条边的路由函数是 assign_worker，它会动态返回一堆 Send
builder.add_conditional_edges(START, assign_worker)

# 所有 processor 节点执行完毕后，都指向 END
builder.add_edge("processor", END)

# 5. 编译并运行
graph = builder.compile()

# 输入一个有 3 个文件的列表
result = graph.invoke({"files": ["a.txt", "b.txt", "c.txt"], "results": []})
print(result)
```

**运行结果**
```text
{
    'files': ['a.txt', 'b.txt', 'c.txt'],
    'results': ['已处理: a.txt', '已处理: b.txt', '已处理: c.txt']
}
```

---

**❓ 为什么这里必须用 `Send`？不用行不行？**
你可能会想：“我能不能不用 `Send`，而是直接在 `assign_worker` 节点里写一个循环，依次调用 `processor`？”

**不行。** 原因在于：

*   如果用普通循环，那是**串行**的，一个一个处理，很慢。
*   `Send` 的强大之处在于，它告诉 LangGraph：“这些任务都是**相互独立**的，请把它们**并行执行**。” LangGraph 会充分利用资源，同时启动多个 `processor` 实例来干活。

---

## 三、图构建器(StateGraph)
### 3.1 介绍
`StateGraph` 是 LangGraph 的核心构建器，用于创建**有状态**、**多步骤**的工作流图。它的核心思想是，图中的各个**节点（Node）** 通过读写一个共享的**状态（State）** 对象来进行通信。

`StateGraph` 本身是一个**构建器（Builder）**，不能直接执行。你需要先通过 `add_node`、`add_edge` 等方法定义图的结构，最后调用 `.compile()` 方法将其编译成一个可执行的图。

**构造函数**

创建 `StateGraph` 实例时，最重要的参数是 `state_schema`，它定义了 State 的结构。

**Python**
```python
from langgraph.graph import StateGraph
from typing import TypedDict

# 1. 使用 TypedDict 定义状态结构
class MyState(TypedDict):
    counter: int
    messages: list

# 2. 创建 StateGraph 实例
builder = StateGraph(state_schema=MyState)
```

**构造函数参数详解 (Python版)**

| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| **`state_schema`** | `Type[StateT]` | **(必填)** 定义整个图的状态结构。可以是 `TypedDict`、Pydantic `BaseModel` 或 `dataclass`。 |
| `context_schema` | `Type[ContextT]` | **可选**。定义运行时上下文（如 `user_id`, `db_conn` 等不变数据）的结构。 |
| `input_schema` | `Type[InputT]` | **可选**。覆盖图的输入模式，用于校验或类型提示。 |
| `output_schema` | `Type[OutputT]` | **可选**。覆盖图的输出模式，用于校验或类型提示。 |

**主要方法(StateGraph)*
`add_node`：添加节点
`add_edge`：添加普通边
`add_conditional_edges`：添加条件边

### 3.2 compile（编译图）

- **作用**：将构建好的 `StateGraph` 编译成一个可执行的图。
- **返回值**：一个 `CompiledStateGraph` 对象。

**示例**:
```python
# 编译图，可在此处配置检查点等
app = builder.compile()
```

**编译后的执行**

编译后得到的 `CompiledStateGraph` 对象支持多种执行方法：

| 方法 | 描述 |
| :--- | :--- |
| **`.invoke(input)`** | 同步执行整个图，并返回最终状态。 |
| **`.ainvoke(input)`** | 异步执行 `invoke`。 |
| **`.stream(input)`** | 同步流式执行，逐步返回状态更新。 |
| **`.astream(input)`** | 异步执行 `stream`。 |

**执行示例 (Python)**:
```python
# 调用图，传入初始状态
final_state = app.invoke({"counter": 0, "messages": []})
print(final_state) # 输出: {'counter': 1, 'messages': []}
```

### 3.3 完整代码示例

下面是一个完整的 Python 示例，展示了一个包含条件循环的计数器图。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义状态
class CounterState(TypedDict):
    counter: int
    max_count: int

# 2. 定义节点
def increment(state: CounterState) -> dict:
    """计数器加1"""
    return {"counter": state["counter"] + 1}

def should_continue(state: CounterState) -> str:
    """判断是否继续增加"""
    if state["counter"] < state["max_count"]:
        return "continue"
    else:
        return "end"

# 3. 构建图
builder = StateGraph(state_schema=CounterState)

# 添加节点
builder.add_node("increment", increment)

# 添加边：从 START 到 increment
builder.add_edge(START, "increment")

# 添加条件边：从 increment 根据 should_continue 的结果跳转
builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "continue": "increment",  # 形成循环
        "end": END
    }
)

# 4. 编译图
app = builder.compile()

# 5. 执行图
initial_state = {"counter": 0, "max_count": 3}
final_state = app.invoke(initial_state)
print(final_state)  # 输出: {'counter': 3, 'max_count': 3}
```












