
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



