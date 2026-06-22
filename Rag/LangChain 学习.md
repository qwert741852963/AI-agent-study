
# LangChain 核心组件学习笔记

## 一、LangChain介绍

### 1.1 简介
LangChain 由 Harrison Chase 创建于 2022 年 10 月，它是围绕 LLMs（大语言模型）建立的一个框架。LangChain 自身并不开发 LLMs，它的核心理念是为各种 LLMs 实现通用的接口，把 LLMs 相关的组件“链接”在一起，简化 LLMs 应用的开发难度，方便开发者快速地开发复杂的 LLMs 应用。

### 1.2 LangChain 主要功能
- **Prompts**：优化提示词（提示词工程）
- **Models**：调用各类模型
- **History**：管理会话历史记录（记忆）
- **Indexes**：管理和分析各类文档
- **Chains**：构建功能的执行链条
- **Agent**：构建智能体

---

## 二、Models 组件

### 2.1 输出结果方式（`invoke` 和 `stream`）

- **`invoke` 方法**：一次性返回完整结果  
- **`stream` 方法**：逐段返回结果，流式输出

#### `invoke` 方法示例
```python
# 以通义千问为例：
from langchain_community.llms.tongyi import Tongyi

# 实例化模型
llm = Tongyi(model='qwen-max')
# 模型推理
res = llm.invoke("帮我讲个笑话吧")
print(res)
```

#### `stream` 方法示例
```python
# langchain_community
from langchain_community.llms.tongyi import Tongyi
# 不用 qwen3-max，因为 qwen3-max 是聊天模型，qwen-max 是大语言模型
model = Tongyi(model="qwen-max")

# 调用 stream 向模型提问
res = model.stream(input="你是谁呀能做什么？")
for chunk in res:
    print(chunk, end="", flush=True)
```

---

### 2.2 Chat Models（聊天模型）

聊天消息包含下面几种类型，使用时需要按照约定传入合适的值：

- **AIMessage**：AI 输出的消息（OpenAI 库中的 `assistant` 角色）
- **HumanMessage**：用户消息（OpenAI 库中的 `user` 角色）
- **SystemMessage**：系统消息，用于指定模型角色、背景或输出格式（OpenAI 库中的 `system` 角色）

#### 2.2.1 单独使用 HumanMessage
```python
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

# 初始化模型
chat = ChatTongyi(model="qwen3-max")
# 准备消息列表
messages = [
    HumanMessage(content="给我写一首唐诗")
]
# 流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)
```

#### 2.2.2 使用 SystemMessage + HumanMessage
```python
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化模型
chat = ChatTongyi(model="qwen3-max")
# 准备消息列表
messages = [
    SystemMessage(content="你是一名来自边塞的诗人"),
    HumanMessage(content="给我写一首唐诗")
]
# 流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)
```

#### 2.2.3 使用 SystemMessage + HumanMessage + AIMessage
```python
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 初始化模型
chat = ChatTongyi(model="qwen3-max")
# 准备消息列表
messages = [
    SystemMessage(content="你是一名来自边塞的诗人"),
    HumanMessage(content="给我写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="给予你上一首的格式，再来一首")
]
# 流式输出
for chunk in chat.stream(input=messages):
    print(chunk.content, end="", flush=True)
```

#### 2.2.4 Models 简写形式

**使用类对象的方式：**
```python
model = ChatTongyi(model="qwen3-max")
messages = [
    SystemMessage(content="你是一个边塞诗人。"),
    HumanMessage(content="写一首唐诗"),
    AIMessage(content="锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    HumanMessage(content="按照你上一个回复的格式，再写一首唐诗。")
]
for chunk in model.stream(input=messages):
    print(chunk.content, end="", flush=True)
```

**简写形式（元组方式）：**
```python
model = ChatTongyi(model="qwen3-max")
messages = [
    ("system", "你是一个边塞诗人。"),
    ("human", "写一首唐诗"),
    ("ai", "锄禾日当午，汗滴禾下土，谁知盘中餐，粒粒皆辛苦。"),
    ("human", "按照你上一个回复的格式，再写一首唐诗。")
]
for chunk in model.stream(input=messages):
    print(chunk.content, end="", flush=True)
```

> **区别与优势**：  
> - 类对象方式：静态，直接得到 Message 类对象。  
> - 简写形式：动态，运行时由 LangChain 内部转换为 Message 对象，支持占位符 `{变量}` 在运行时填充具体值（后续在提示词模板中使用）。

---

### 2.3 Embeddings Models（文本嵌入模型）

嵌入模型将字符串作为输入，返回一个浮点数列表（向量），用于文本向量化。

**阿里云千问模型访问方式：**
```python
from langchain_community.embeddings import DashScopeEmbeddings

# 初始化嵌入模型对象，默认使用 text-embedding-v1
embed = DashScopeEmbeddings()
# 测试
print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(['我喜欢你', '我稀饭你', '晚上吃啥']))
```

**本地 Ollama 模型访问方式：**
```python
from langchain_ollama import OllamaEmbeddings

# 初始化嵌入模型对象，默认使用 qwen3-embedding
embed = OllamaEmbeddings(model="qwen3-embedding")
# 测试
print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(['我喜欢你', '我稀饭你', '晚上吃啥']))
```

---

## 三、Prompts（提示词）组件

### 3.1 通用 PromptTemplate

通用提示词模板，支持动态注入信息。

**标准写法：**
```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 帮忙起名字，请简略回答。"
)
# 变量注入，生成提示词文本
prompt_text = prompt_template.format(lastname="张", gender="女儿")

model = Tongyi(model="qwen-max")
res = model.invoke(input=prompt_text)
print(res)
```

**基于 Chain 链的写法：**
```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 帮忙起名字，请简略回答。"
)
model = Tongyi(model="qwen-max")
# 生成链
chain = prompt_template | model
# 基于链，调用模型获取结果
res = chain.invoke(input={"lastname": "曹", "gender": "女儿"})
print(res)
```

---

### 3.2 FewShotPromptTemplate

支持基于模板注入任意数量的示例信息。

**构造函数参数：**
```python
from langchain_core.prompts import FewShotPromptTemplate

FewShotPromptTemplate(
    examples=None,          # 示例数据，list，内套字典
    example_prompt=None,    # 示例数据的提示词模板
    prefix=None,            # 示例数据前的内容
    suffix=None,            # 示例数据后的内容
    input_variables=None    # 注入的变量列表
)
```

**组装并获得最终提示词：**
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_template = PromptTemplate.from_template("单词:{word}, 反义词:{antonym}")
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]
few_shot_prompt = FewShotPromptTemplate(
    example_prompt=example_template,
    examples=example_data,
    prefix="给出给定词的反义词，有如下示例：\n",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=['input_word']
)
# 获得最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()
print(prompt_text)
```

**调用模型获得结果：**
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

example_template = PromptTemplate(
    input_variables=['word', 'antonym'],
    template="word: {word}, antonym: {antonym}"
)
example_data = [
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]
few_shot_prompt = FewShotPromptTemplate(
    examples=example_data,
    example_prompt=example_template,
    prefix="给出给定词的反义词，有如下示例：\n",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=['input_word']
)
# 获得最终提示词
prompt_text = few_shot_prompt.invoke(input={"input_word": "左"}).to_string()

model = ChatTongyi(model="qwen3-max")
for chunk in model.stream(input=prompt_text):
    print(chunk.content, end="", flush=True)
```

---

### 3.3 模板类的 `format` 和 `invoke` 方法

- **`format` 方法**：`format(**kwargs)` → 返回字符串  
- **`invoke` 方法**：`invoke(input: dict)` → 返回 `PromptValue` 对象，可进一步调用 `.to_string()` 或 `.to_messages()`

---

### 3.4 ChatPromptTemplate

支持注入任意数量的历史会话信息。通过 `from_messages` 方法，从列表中获取多轮次会话作为聊天的基础模板。

**基本结构：**
```python
from langchain_core.prompts import ChatPromptTemplate

ChatPromptTemplate.from_messages(
    [
        ("system", "........"),
        ("ai", "........"),
        # ...
        ("human", "........")
    ]
)
```

**动态注入历史会话（使用 `MessagesPlaceholder`）：**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "........"),
        ("ai", "........"),
        MessagesPlaceholder("history"),   # history 作为占位 key
        ("human", "........")
    ]
)

history_data = [
    ("human", "..."),
    ("ai", "..."),
    ("human", "..."),
    ("ai", "...")
]
# 使用 invoke 动态注入历史会话记录（format 无法注入）
prompt_value = chat_template.invoke({"history": history_data})
```

**结合聊天模型进行 Few-shot 提示（求反义词）：**
```python
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"},
]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "给出每个单词的反义词"),
        MessagesPlaceholder("history"),   # history 占位
        ("human", "{question}")
    ]
)

model = ChatTongyi(model="qwen3-max")
# StrOutputParser 提取结果文本内容，剔除元数据
chain = prompt | model | StrOutputParser()

# 无历史会话的提问
for chunk in chain.stream(input={"history": [], "question": "粗"}):
    print(chunk)

print("*" * 20)

# 带有历史的提问（使用 HumanMessage 和 AIMessage 封装）
history = [
    HumanMessage(content="开心"),
    AIMessage(content="难过"),
    HumanMessage(content="高"),
    AIMessage(content="矮")
]
# 也可用元组简写：
# history = [("human", "开心"), ("ai", "难过"), ("human", "高"), ("ai", "矮")]

for chunk in chain.stream(input={"history": history, "question": "粗"}):
    print(chunk)
```
```
