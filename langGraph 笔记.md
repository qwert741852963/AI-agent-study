
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

#### 1.1.2 更新状态（Reducer 的实战）

节点函数接收当前 `state` 作为参数，并**返回一个部分状态更新（Partial State）**，而不是直接修改 `state`。

**示例 1：无 Reducer（直接覆盖）**

如果状态字段没有 Reducer，节点的返回值会直接覆盖该字段的旧值。

```python
from langgraph.graph import StateGraph, START, END

# 1. 定义状态
class State(TypedDict):
    current_output: str

# 2. 定义节点
def node_a(state: State):
    # 返回的 current_output 会直接覆盖状态中的旧值
    return {"current_output": "Hello from Node A"}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", END)
graph = builder.compile()

# 4. 执行
initial_state = {"current_output": "initial"}
final_state = graph.invoke(initial_state)
print(final_state) # 输出: {'current_output': 'Hello from Node A'}
```

**示例 2：使用 Reducer（合并更新）**

为字段指定 Reducer，可以实现复杂的合并逻辑，如消息的累加。

```python
from typing import List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, AIMessage

# 1. 定义状态，使用 Annotated 指定 reducer
def add_messages(left: List[AnyMessage], right: List[AnyMessage]) -> List[AnyMessage]:
    """Reducer 函数：将消息列表合并"""
    return left + right

class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]  # 关键：指定 reducer

# 2. 定义节点
def node_a(state: State):
    # 返回的消息列表会通过 add_messages reducer 与现有列表合并
    return {"messages": [AIMessage(content="Hello from Node A")]}

def node_b(state: State):
    return {"messages": [AIMessage(content="Hello from Node B")]}

# 3. 构建图
builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b") # node_a -> node_b
builder.add_edge("node_b", END)
graph = builder.compile()

# 4. 执行
initial_state = {"messages": []}
final_state = graph.invoke(initial_state)
print(final_state["messages"])
# 输出: [AIMessage(content='Hello from Node A'), AIMessage(content='Hello from Node B')]
```

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

## 1.2.1 节点的定义方式

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
















