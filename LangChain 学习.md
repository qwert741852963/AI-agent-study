
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



## 四、Chains 组件

### 4.1 链的基础使用
「将组件串联，上一个组件的输出作为下一个组件的输入」是 LangChain 链（尤其是 `|` 管道链）的核心工作原理，这也是链式调用的核心价值：实现数据的自动化流转与组件的协同工作。

```python
chain = prompt_template | model
```

**核心前提**：即 `Runnable` 子类对象才能入链（以及 `Callable`、`Mapping` 接口子类对象也可加入）。我们目前所学习到的组件，均是 `Runnable` 接口的子类。

通过 `|` 链接提示词模板对象和模型对象，返回值 `chain` 对象是 `RunnableSerializable` 对象，是 `Runnable` 接口的直接子类，也是绝大多数组件的父类。通过 `invoke` 或 `stream` 进行阻塞执行或流式执行。

组成的链在执行上有：**上一个组件的输出作为下一个组件的输入**的特性。

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables.base import RunnableSerializable

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗，无需额外输出"),
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

model = ChatTongyi(model="qwen3-max")
chain: RunnableSerializable = chat_prompt_template | model
print(type(chain))

# Runnable 接口，invoke 执行
res = chain.invoke({"history": history_data})
print(res.content)

# Runnable 接口，stream 执行
for chunk in chain.stream({"history": history_data}):
    print(chunk.content, end="", flush=True)
```

> **小结**：LangChain 中链是一种将各个组件串联在一起，按顺序执行，前一个组件的输出作为下一个组件的输入。可以通过 `|` 符号来让各个组件形成链，成链的各个组件需是 `Runnable` 接口的子类。形成的链是 `RunnableSerializable` 对象（`Runnable` 接口子类），可通过链调用 `invoke` 或 `stream` 触发整个链条的执行。

---

## 五、LangChain 组件：输出解析器

### 5.1 StrOutputParser

有如下代码，想要以第一次模型的输出结果，第二次去询问模型，会报错：

```python
model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}, 刚生了{gender}，请起名，仅告知名字无需其它内容"
)
chain = prompt | model | model
res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res.content)
```

运行报错（`ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.`）

错误的主要原因是：`prompt` 的结果是 `PromptValue` 类型，输入给了 `model`，`model` 的输出结果是 `AIMessage`。模型（`ChatTongyi`）源码中关于 `invoke` 方法明确指定了 `input` 的类型。

需要做类型转换，可以借助 LangChain 内置的解析器 `StrOutputParser`：

```python
parser = StrOutputParser()
chain = prompt | model | parser | model
```

`StrOutputParser` 是 LangChain 内置的简单字符串解析器，可以将 `AIMessage` 类型转换为基础字符串，可以加入 `chain` 作为组件存在（`Runnable` 接口子类）。

> **小结**：`StrOutputParser` 是 LangChain 内置的简单字符串解析器，可以将 `AIMessage` 解析为简单的字符串，符合了模型 `invoke` 方法要求（可传入字符串，不接收 `AIMessage` 类型）。它是 `Runnable` 接口的子类，可以加入链。

---

### 5.2 JsonOutputParser 与多模型执行链

在前面我们完成了这样的需求去构建多模型链，不过这种做法并不标准，因为上一个模型的输出没有被处理就输入下一个模型。正常情况下我们应该有如下处理逻辑：上一个模型的输出结果，应该作为提示词模板的输入，构建下一个提示词，用来二次调用模型。

```python
chain = prompt | model | parser | model | parser
# invoke｜stream 初始输入 → 提示词模板 → 模型 → 数据处理 → 提示词模板 → 模型 → 解析器 → 结果
```

将模型输出的 `AIMessage` → 转为字典 → 注入第二个提示词模板中，形成新的提示词（`PromptValue` 对象）。  
`StrOutputParser` 不满足（`AIMessage` → `Str`），更换 `JsonOutputParser`（`AIMessage` → `Dict(JSON)`）。

**JsonOutputParser 完成多模型链：**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
json_parser = JsonOutputParser()
model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，并封装到JSON格式返回给我，"
    "要求key是name，value就是起的名字。请严格遵守格式要求"
)
second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

chain = first_prompt | model | json_parser | second_prompt | model | str_parser
res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
print(type(res))
```

> **注意**：在构建链的时候要注意整体兼容性，注意前后组件的输入和输出要求：
> - 模型输入：`PromptValue` 或字符串或序列（`BaseMessage`、`list`、`tuple`、`str`、`dict`）
> - 模型输出：`AIMessage`
> - 提示词模板输入：要求是字典
> - 提示词模板输出：`PromptValue` 对象
> - `StrOutputParser`：`AIMessage` 输入、`str` 输出
> - `JsonOutputParser`：`AIMessage` 输入、`dict` 输出

---

### 5.3 RunnableLambda 与函数加入链

前文我们根据 `JsonOutputParser` 完成了多模型执行链条的构建。除了 `JsonOutputParser` 这类固定功能的解析器之外，我们也可以自己编写 Lambda 匿名函数来完成自定义逻辑的数据转换，想怎么转换就怎么转换，更自由。

想要完成这个功能，可以基于 `RunnableLambda` 类实现。`RunnableLambda` 类是 LangChain 内置的，将普通函数等转换为 `Runnable` 接口实例，方便自定义函数加入 chain。

**语法**：`RunnableLambda(函数对象或 lambda 匿名函数)`

**使用 RunnableLambda：**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

str_parser = StrOutputParser()
my_func = RunnableLambda(lambda ai_msg: {"name": ai_msg.content})
model = ChatTongyi(model="qwen3-max")

first_prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}，刚生了{gender}，请起名，仅告知我名字，不要额外信息"
)
second_prompt = PromptTemplate.from_template(
    "姓名{name}，请帮我解析含义。"
)

chain = first_prompt | model | my_func | second_prompt | model | str_parser
res: str = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res)
print(type(res))
```

**函数直接入链（更简洁）：**

```python
chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser
```

---



## 五、文本分割器(Text Splitters)

### 1. `CharacterTextSplitter` (字符文本分割器)

这是最基础的分割器，它会根据一个**指定的字符序列**（默认是 `"\n\n"`）来分割文本，并按**字符数**计算块大小。

```python
from langchain_text_splitters import CharacterTextSplitter

# 1. 加载你的文档
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# 2. 创建分割器实例
text_splitter = CharacterTextSplitter(
    separator="\n\n",        # 指定分隔符，默认为 "\n\n"
    chunk_size=1000,         # 每个块的最大字符数
    chunk_overlap=200,       # 块之间的重叠字符数
    length_function=len,     # 计算文本长度的函数
    is_separator_regex=False # 是否将分隔符解释为正则表达式
)

# 3. 执行分割

# 方法一：返回 LangChain Document 对象列表（可携带元数据）
documents = text_splitter.create_documents([state_of_the_union])
print(documents[0])

# 方法二：直接返回纯文本字符串列表
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```

### 2. `RecursiveCharacterTextSplitter` (递归字符文本分割器)

这是**官方推荐**的通用分割器。它会按优先级顺序尝试用一个**字符列表**（默认为 `["\n\n", "\n", " ", ""]`）进行分割，优先保持段落、句子等较大语义单元的完整。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 加载你的文档
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# 2. 创建分割器实例
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,          # 每个块的最大字符数
    chunk_overlap=20,        # 块之间的重叠字符数
    length_function=len,     # 计算文本长度的函数
    is_separator_regex=False # 是否将分隔符列表解释为正则表达式
)

# 3. 执行分割
texts = text_splitter.split_text(state_of_the_union)
print(texts[0])
```

### 3. 代码文本分割器 (Code Text Splitter)

这是 `RecursiveCharacterTextSplitter` 的一个特化版本，针对特定编程语言提供了优化的分隔符列表，能更好地保持代码结构。

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python 代码示例
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")
# Call the function
hello_world()
"""

# 创建针对 Python 优化的分割器
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
print(python_docs)

# JavaScript 代码示例
JS_CODE = """
function helloWorld() {
  console.log("Hello, World!");
}
// Call the function
helloWorld();
"""
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=60,
    chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
print(js_docs)
```

> **支持的语言**：`cpp`, `go`, `java`, `kotlin`, `js`, `ts`, `php`, `python`, `ruby`, `rust`, `scala`, `swift`, `html`, `csharp`, `cobol`, `c`, `lua`, `perl`, `haskell` 等。

### 4. `MarkdownHeaderTextSplitter` (Markdown标题分割器)

它能感知 Markdown 的结构，根据指定的标题层级（如 `#`, `##`）来分割文档，并将标题作为元数据保留。

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_document = """
# Foo

## Bar
Hi this is Jim

Hi this is Joe

### Boo
Hi this is Lance

## Baz
Hi this is Molly
"""

# 指定要作为分割依据的标题层级及其在元数据中的名称
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# 创建分割器实例
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

# 执行分割
md_header_splits = markdown_splitter.split_text(markdown_document)

for doc in md_header_splits:
    print(doc)
# 输出会显示每个块的内容及其对应的标题层级元数据
```

### 5. `HTMLHeaderTextSplitter` (HTML标题分割器)

与 Markdown 分割器类似，它根据 HTML 的标题标签（如 `<h1>`, `<h2>`）来分割内容，并生成带有层级元数据的 `Document` 对象。

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_content = """
<html>
  <body>
    <h1>Introduction</h1>
    <p>Welcome to the introduction section.</p>
    <h2>Background</h2>
    <p>Some background details here.</p>
    <h1>Conclusion</h1>
    <p>Final thoughts.</p>
  </body>
</html>
"""

# 指定要分割的标题标签及对应的元数据名称
headers_to_split_on = [("h1", "Main Topic"), ("h2", "Sub Topic")]

# 创建分割器实例
splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    return_each_element=False  # 设为 True 会为每个 HTML 元素创建一个 Document
)

# 执行分割
documents = splitter.split_text(html_content)

for doc in documents:
    print(doc)
# 输出会显示每个块的内容及其所属的标题层级元数据
```

### 6. `RecursiveJsonSplitter` (递归JSON分割器)

这个分割器用于处理大型 JSON 数据。它会深度优先遍历 JSON，将其拆分成更小的、结构化的 JSON 块，并允许控制块的大小。

```python
import json
from langchain_text_splitters import RecursiveJsonSplitter

# 示例：一个大型的 JSON 数据（这里用字典表示）
json_data = {
    "openapi": "3.1.0",
    "info": {"title": "LangSmith", "version": "0.1.0"},
    "servers": [{"url": "https://api.smith.langchain.com"}],
    "paths": {
        "/api/v1/sessions/{session_id}": {
            "get": {
                "tags": ["tracer-sessions"],
                "summary": "Read Tracer Session"
            }
        }
    }
}

# 创建分割器实例，指定最大块大小
splitter = RecursiveJsonSplitter(max_chunk_size=300)

# 方法一：获取 JSON 块（字典对象）
json_chunks = splitter.split_json(json_data=json_data)
for chunk in json_chunks[:2]:
    print(json.dumps(chunk, indent=2))

# 方法二：获取 LangChain Document 对象
docs = splitter.create_documents(texts=[json_data])
for doc in docs[:2]:
    print(doc)

# 方法三：直接获取文本字符串
texts = splitter.split_text(json_data=json_data)
print(texts[0])
```

### 7、`TokenTextSplitter` - 基于 Token 数量的分割器

按 **Token 数量** 而非字符数来分割文本，适合需要精确控制 Token 消耗的场景。

```python
from langchain_text_splitters import TokenTextSplitter

# 加载文档
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

# 创建分割器实例
text_splitter = TokenTextSplitter(
    chunk_size=50,          # 每个块的最大 Token 数
    chunk_overlap=10,       # 块之间的重叠 Token 数
    encoding_name="cl100k_base"  # 使用的 Tokenizer 编码，默认 "gpt2"
)

# 执行分割
texts = text_splitter.split_text(state_of_the_union)
print(f"分割后的块数: {len(texts)}")
print(texts[0])
```






### 8、上诉分割器适用场景

| 分割器 | 适用场景 |
| :--- | :--- |
| **`RecursiveCharacterTextSplitter`** | **通用文本的首选**，平衡了语义完整性和块大小控制。 |
| **`CharacterTextSplitter`** | 需要非常简单的、基于固定字符序列的分割时。 |
| **代码文本分割器** | 处理 Python、JavaScript 等编程语言的代码。 |
| **`MarkdownHeaderTextSplitter`** | 处理 Markdown 格式的文档，需要保留其标题结构。 |
| **`HTMLHeaderTextSplitter`** | 处理 HTML 格式的文档，需要保留其标题结构。 |
| **`RecursiveJsonSplitter`** | 处理大型、嵌套的 JSON 数据。 |


### 9、其他分割器

| 分割器名称 | 描述 |
|------------|------|
| `TokenTextSplitter` | 基于 Token 数量拆分，对受 Token 限制的 LLM 更精确。 |
| `HTMLHeaderTextSplitter` | 根据 HTML 标题标签（`<h1>`, `<h2>` 等）拆分，保留层级元数据。 |
| `HTMLSectionSplitter` | 基于指定的 HTML 标签或字体大小将文档拆分为更大的区块。 |
| `HTMLSemanticPreservingSplitter` | 拆分时保留表格、列表等 HTML 元素的语义结构，确保它们不被切断。 |
| `JSFrameworkTextSplitter` | 专门用于处理 React（JSX）、Vue 和 Svelte 等前端框架的代码。 |
| `ExperimentalMarkdownSyntaxTextSplitter` | 实验性的分割器，旨在处理 Markdown 语法时保留原始空白字符并提取结构化元数据。 |




## 六、Memory 组件
### 6.1 短期与长期记忆区别
简单来说，短期记忆是AI在单次对话中的“工作记忆”，保证对话连贯；长期记忆则是AI的“个人数据库”，让AI能“记住”用户，提供个性化服务。
在实际应用中，两者常常结合使用。比如，先用长期记忆检索用户偏好和背景信息，再将这些信息连同当前的短期记忆（对话历史）一起，作为上下文提供给大模型，从而生成最贴切的回复。
**场景一：健康/饮食助手**
短期记忆：根据用户今天输入的“早餐吃了两个鸡蛋”，推荐合适的午餐。
长期记忆：记住用户“对花生过敏”和“正在控制碳水摄入”，在所有推荐中自动规避风险食物。

**场景二：编程助手**
短期记忆：在当前对话中，记得用户刚说“用Python”，因此后续代码示例都用Python给出。
长期记忆：记住用户“偏好简洁的代码风格”和“主要做数据分析”，主动推荐pandas库，并使用其更简洁的API写法。

### 6.2 短期记忆 (Short-term Memory)
短期记忆的核心机制是 **检查点（Checkpointer）**。它可以理解为在对话的每一步都自动“存档”，确保应用能在任何时候恢复状态，实现连贯的多轮对话。
在LangGraph中，短期记忆通过“检查点”实现。你可以把`thread_id`想象成一个对话的房间号。LangGraph会以这个房间号为标识，把每次对话的状态（包括消息历史、当前步骤等）都存成一份“存档”。

*   **工作流程**：当Agent开始处理一个请求，它会读取该`thread_id`对应的最新“存档”来“回忆”起之前的对话。当处理完一个步骤（如调用工具）或一次完整的对话后，新的状态又会被保存为一份新“存档”。

| 检查点类型 (Checkpointer) | 适用场景 | 持久化 | 特点 |
| :--- | :--- | :--- | :--- |
| **`InMemorySaver`** | 本地开发、单元测试 | ❌ 内存 | 简单快速，无外部依赖 |
| **`SqliteSaver`** | 小型项目、单机应用 | ✅ 本地文件 | 轻量级，无需额外服务 |
| **`PostgresSaver`** | **生产环境（推荐）** | ✅ 数据库 | 高可靠、功能完整，适合关键任务 |
| **`MongoDBSaver`** | 生产环境 | ✅ 数据库 | 适合已有MongoDB技术栈的团队 |
| **`RedisSaver`** | 生产环境（高性能） | ✅ 内存数据库 | 速度极快，适合低延迟场景 |

#### 1. 开发与测试 (MemorySaver、InMemorySaver)

这些实现将数据保存在内存中，进程重启后数据会丢失，**不适合用于生产环境**。

*   **`MemorySaver` / `InMemorySaver`**：最基础的检查点实现，适合快速原型开发和单元测试。
    *   **特点**：简单、快速，无需任何外部依赖。
    *   **适用场景**：本地开发、调试、编写单元测试。

    ```python
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import StateGraph, MessagesState

    # 1. 创建内存检查点
    checkpointer = InMemorySaver()

    # 2. 构建你的图 (假设已经定义好了 builder)
    # builder = StateGraph(MessagesState) ...
    
    # 3. 编译图时传入 checkpointer
    graph = builder.compile(checkpointer=checkpointer)

    # 4. 调用时指定 thread_id
    config = {"configurable": {"thread_id": "user_123_session_1"}}
    graph.invoke({"messages": [("user", "你好，我叫小明。")]}, config=config)
    ```
**具体例子**
```python
# 安装依赖：pip install langgraph langchain-openai

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# 1. 创建真实的大模型
model = ChatOpenAI(model="gpt-4o-mini", api_key="你的API密钥")

# 2. 定义AI节点
def ai_node(state: MessagesState):
    # 调用大模型，它会自动理解上下文
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# 3. 构建工作流（和之前一样）
builder = StateGraph(MessagesState)
builder.add_node("ai_node", ai_node)
builder.add_edge(START, "ai_node")

# 4. 添加记忆（和之前一样）
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 5. 对话
config = {"configurable": {"thread_id": "room_001"}}

# 第一轮
graph.invoke(
    {"messages": [{"role": "user", "content": "你好，我叫小明"}]},
    config=config
)

# 第二轮
result = graph.invoke(
    {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
    config=config
)
print(result["messages"][-1].content) 
# 输出：你叫小明。 ✅ 这才真正记住了！
```
InMemorySaver()	创建一个“临时记事本”，用来记对话
compile(checkpointer=xxx)	把“记事本”交给AI，让它有记性
thread_id	为不同聊天室贴标签，相同标签的对话会被记在一起
程序重启	用 InMemorySaver 的话，所有记忆都会丢失


#### 2. 生产环境 (Production)

在生产环境，必须使用由数据库支持的检查点，以实现真正的持久化。
##### PostgresSaver
*   **`PostgresSaver`**：将检查点存储在PostgreSQL数据库中。
    *   **特点**：生产环境最常用的方案，可靠性高，支持完整的历史记录。
    *   **适用场景**：需要高可靠性和数据完整性的正式线上服务。
    *   **注意**：首次使用前需调用`checkpointer.setup()`创建必要的数据库表。

    ```python
    from langgraph.checkpoint.postgres import PostgresSaver

    DB_URI = "postgresql://user:pass@localhost:5432/mydb"
    
    # 使用上下文管理器，自动处理连接
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        # 首次使用需要执行 setup 创建表
        # checkpointer.setup() 
        
        # 编译图时传入 checkpointer
        graph = builder.compile(checkpointer=checkpointer)
        # ... 后续调用与 InMemorySaver 类似
    ```
    
    **具体例子**
    
```python
# 1. 导入所需库
import os
from langchain.chat_models import init_chat_model  # 用于初始化各种模型[reference:17]
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver

# 2. 配置你的模型 (以OpenAI为例)
# 确保环境变量 OPENAI_API_KEY 已设置
# 或者在代码中直接设置: os.environ["OPENAI_API_KEY"] = "your-api-key"
model = init_chat_model(model="gpt-4o-mini") # [reference:18]

# 3. 定义PostgreSQL连接字符串
DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

# 4. 定义AI节点的逻辑
def call_model(state: MessagesState):
    """接收当前状态（包含所有历史消息），调用模型并返回回复"""
    # state["messages"] 包含了整个对话历史
    response = model.invoke(state["messages"]) # [reference:19]
    # 返回的消息会被自动添加到状态中
    return {"messages": response}

# 5. 构建工作流 (Graph)
builder = StateGraph(MessagesState) # [reference:20]
builder.add_node("call_model", call_model) # [reference:21]
builder.add_edge(START, "call_model")

# 6. 【核心】使用 PostgresSaver 编译工作流
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # 【重要】首次运行时，取消下面一行的注释来创建数据库表
    # checkpointer.setup() # [reference:22]
    
    # 编译图时传入 checkpointer，使应用获得持久化的短期记忆能力
    graph = builder.compile(checkpointer=checkpointer) # [reference:23]

    # 7. 开始对话
    # 配置 thread_id，用于标识不同的对话线程
    config = {"configurable": {"thread_id": "user_123_session_1"}} # [reference:24]

    print("=" * 50)
    print("第一轮对话")
    print("=" * 50)
    # 用户的第一条消息
    user_input = {"messages": [{"role": "user", "content": "你好，我叫小明。"}]}
    # 通过 stream 方法获取模型回复（流式输出）
    for chunk in graph.stream(user_input, config, stream_mode="values"): # [reference:25]
        chunk["messages"][-1].pretty_print()

    print("\n" + "=" * 50)
    print("第二轮对话 (AI应该还记得你的名字)")
    print("=" * 50)
    # 用户的第二条消息
    user_input = {"messages": [{"role": "user", "content": "你还记得我叫什么名字吗？"}]}
    for chunk in graph.stream(user_input, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

# 程序结束后，with语句会自动关闭数据库连接
```
##### MongoDBSaver
*   **`MongoDBSaver`**：将检查点存储在MongoDB中。
    *   **特点**：适合已有MongoDB基础设施的团队。
    *   **适用场景**：项目本身基于MongoDB，希望统一技术栈。

    ```python
    from langgraph.checkpoint.mongodb import MongoDBSaver
    from pymongo import MongoClient

    client = MongoClient("mongodb://user:password@localhost:27017")
    checkpointer = MongoDBSaver(client) # 可指定数据库名，默认为 "langgraph"
    
    graph = builder.compile(checkpointer=checkpointer)
    ```
##### RedisSaver
*   **`RedisSaver`**：将检查点存储在Redis中。
    *   **特点**：速度极快，适合对延迟敏感的应用。提供标准版（`RedisSaver`，保存完整历史）和轻量版（`ShallowRedisSaver`，只保留最新检查点）。
    *   **适用场景**：追求高性能、低延迟的对话系统。
    *   **注意**：首次使用前需调用`.setup()`方法创建必要的索引。

    ```python
    from langgraph.checkpoint.redis import RedisSaver

    # 从连接字符串创建
    checkpointer = RedisSaver.from_conn_string("redis://localhost:6379")
    # 首次使用需要 setup
    # checkpointer.setup()
    
    graph = builder.compile(checkpointer=checkpointer)
    ```
##### SqliteSaver
*   **`SqliteSaver`**：将检查点存储在本地SQLite文件中。
    *   **特点**：轻量级，无需单独部署数据库服务。
    *   **适用场景**：单机应用、小型项目或演示。
        ```python
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        # 连接到本地 SQLite 文件
        checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
        graph = builder.compile(checkpointer=checkpointer)
        ```

### 6.3 长期记忆
LangChain的长期记忆（Long-term Memory）是指代理（Agent）能够**在不同对话和会话之间持久存储并回忆信息**的能力。

LangChain的长期记忆建立在 **LangGraph 的存储（Store）** 之上。记忆以 **JSON 文档** 的形式保存，并通过两层结构来组织：

1.  **命名空间（Namespace）**：类似“文件夹”，用于对记忆进行分组。通常会包含用户ID或组织ID等信息，方便管理。
2.  **键（Key）**：类似“文件名”，是每条记忆在该命名空间下的唯一标识符。

LangChain 长期记忆的核心用法是**通过工具（Tool）来读写存储（Store）**。

**LangChain 提供了多种存储后端**：

*   **`InMemoryStore`**：用于开发和测试的**内存存储**，数据不会持久化。
*   **`PostgresStore`**：基于 **PostgreSQL** 的生产环境存储。
*   **`MongoDBStore`**：基于 **MongoDB** 的存储实现。
*   **`CosmosDBStore`**：基于 **Azure Cosmos DB** 的存储。

**通过以下步骤为代理赋予持久记忆能力：**
1.  选择一个存储后端（如 `InMemoryStore`, `PostgresStore`）。
2.  在创建代理时，将该存储实例传入。
3.  在自定义的工具函数中，通过 `runtime.store` 的 `put`、`get` 和 `search` 方法来读写记忆。

**PostgresStore举例**

```python
import os
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.tools import ToolRuntime
from langgraph.store.postgres import PostgresStore
from langchain_openai import ChatOpenAI

# 1. 配置数据库连接
DB_URI = "postgresql://user:pass@localhost:5432/dbname"

# 2. 初始化 LLM
model = ChatOpenAI(model="gpt-4o-mini")
```

2、这是核心部分。我们定义两个工具，分别用于**写入**和**读取**用户的长期记忆。通过 `runtime.store` 对象，工具可以直接与 `PostgresStore` 交互。

```python
# 写入记忆工具：写入用户偏好
@tool
def save_user_preference(input: str, runtime: ToolRuntime) -> str:
    """保存用户的编程语言偏好。输入是用户偏好的语言，例如 'Python'。"""
    # 使用命名空间来隔离不同用户的数据 (这里用 'default-user' 作为示例)
    namespace = ("default-user", "preferences")
    # 将用户的偏好存入 store，key 为 "programming_language"
    runtime.store.put(namespace, "programming_language", {"language": input})
    return f"已成功保存您的偏好：{input}"

# 读取记忆工具：读取用户偏好
@tool
def get_user_preference(runtime: ToolRuntime) -> str:
    """获取用户之前保存的编程语言偏好。"""
    namespace = ("default-user", "preferences")
    # 从 store 中读取数据
    item = runtime.store.get(namespace, "programming_language")
    if item:
        # item.value 包含了存储的字典
        return f"您的编程语言偏好是：{item.value['language']}"
    return "您还没有设置编程语言偏好。"
```

3. 创建带有长期记忆的 Agent
将 `PostgresStore` 实例和定义好的工具传递给 `create_agent` 函数。

```python
# 5. 创建 PostgresStore 并初始化数据库表
# 使用 with 语句管理资源，确保连接正常关闭
with PostgresStore.from_conn_string(DB_URI) as store:
    # 首次运行时，创建必要的数据库表
    store.setup()

    # 6. 创建 Agent
    agent = create_agent(
        model=model,
        tools=[save_user_preference, get_user_preference],
        store=store,  # 将 store 传入 agent
    )

    # 7. 与 Agent 对话
    # 第一轮对话：让 Agent 记住一个偏好
    print("--- 第一轮对话：保存偏好 ---")
    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我喜欢用 Python 编程。"}]}
    )
    # 打印 Agent 的最终回复
    print(result1["messages"][-1].content)

    # 第二轮对话：在新的线程中，让 Agent 回忆偏好
    # 注意：这里没有传入 thread_id，但长期记忆是跨线程的，所以依然能访问到
    print("\n--- 第二轮对话：回忆偏好 ---")
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "你知道我喜欢的编程语言是什么吗？"}]}
    )
    print(result2["messages"][-1].content)
```

4. 运行与输出
执行上述代码，你会看到类似下面的输出：
```
--- 第一轮对话：保存偏好 ---
已成功保存您的偏好：Python

--- 第二轮对话：回忆偏好 ---
您的编程语言偏好是：Python
```
在第二轮对话中，尽管没有传入任何历史消息（即没有短期记忆），但 Agent 依然通过 `get_user_preference` 工具从 `PostgresStore` 中读取到了第一轮保存的偏好，证明了长期记忆的有效性。

**5. 关键点总结**
1.  **`runtime.store` 是桥梁**：所有对长期记忆的读写操作，都通过工具函数中的 `runtime.store` 对象完成。
2.  **命名空间 (`namespace`) 用于数据隔离**：通过包含 `user_id` 的元组（如 `("user_123", "preferences")`）来隔离不同用户的数据，是推荐的最佳实践。
3.  **`store.setup()` 只需运行一次**：它的作用是创建必要的数据库表，可以在应用启动时执行。
4.  **生产环境使用 `PostgresStore`**：示例中的 `InMemoryStore` 仅用于测试，生产环境务必使用 `PostgresStore` 等持久化存储。

---

## 七、LangChain 组件：Document Loaders（文档加载器）

文档加载器提供了一套标准接口，用于将不同来源（如 CSV、PDF 或 JSON 等）的数据读取为 LangChain 的文档格式。这确保了无论数据来源如何，都能对其进行一致性处理。

文档加载器（内置或自行实现）需实现 `BaseLoader` 接口。

**`Document` 类其核心记录了：**
- `page_content`：文档内容
- `metadata`：文档元数据（字典）

不同的文档加载器可能定义了不同的参数，但是其都实现了统一的接口（方法）：
- `load()`：一次性加载全部文档
- `lazy_load()`：延迟流式传输文档，对大型数据集很有用，避免内存溢出

---

### 7.1 CSVLoader

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./xxx.csv")
data = loader.load()
print(data)
```

**CSVLoader 自定义 CSV 文件的解析和加载：**

```python
loader = CSVLoader(
    file_path="./xxx.csv",
    csv_args={
        "delimiter": ",",      # 指定分隔符
        "quotechar": '"',      # 指定字符串的引号包裹
        # 字段列表（无表头使用，有表头勿用会读取首行做为数据）
        "fieldnames": ["name", "age", "gender"],
    },
)
data = loader.load()
print(data)
```

> **小结**：
> - LangChain 内置了许多种类的文档加载器
> - 文档加载器均继承于 `BaseLoader` 类
> - 返回 `Document` 类型的对象
> - `load` 方法一次性批量加载（返回 `list` 内含 `Document` 对象），如内容过多可能 list 太大，出现内存溢出问题
> - `lazy_load` 方法会得到生成器对象，可用 for 循环依次获取单个 `Document` 对象，适用于大文档避免内存存不下
> - `CSVLoader` 用于加载 CSV 文件，加载成功得到的即 `Document` 对象

---

### 7.2 JSONLoader

`JSONLoader` 用于将 JSON 数据加载为 `Document` 类型对象。使用 `JSONLoader` 需要额外安装：`pip install jq`。jq 是一个跨平台的 json 解析工具，LangChain 底层对 JSON 的解析就是基于 jq 工具实现的。

将 JSON 数据的信息抽取出来，封装为 `Document` 对象，抽取的时候依赖 `jq_schema` 语法。

**常见 jq 语法：**
- `.` 表示整个 JSON 对象（根）
- `[]` 表示数组
- `.name` 表示抽取 `name` 的值
- `.hobby` 表示抽取爱好数组
- `.hobby[1]` 或 `.hobby.[1]` 表示抽取第二个爱好
- `.other.addr` 表示抽取地址

**示例 JSON：**
```json
{
    "name": "周杰轮",
    "age": 11,
    "hobby": ["唱", "跳", "RAP"],
    "other": {
        "addr": "深圳",
        "tel": "12332112321"
    }
}
```

**数组示例：**
```json
[
    {"name": "周杰轮", "age": 11, "gender": "男"},
    {"name": "蔡依临", "age": 12, "gender": "女"},
    {"name": "王力鸿", "age": 11, "gender": "男"}
]
```
- `.[]` 得到 3 个字典
- `.[].name` 表示抽取全部的 `name`，即得到 3 个 `name` 信息

了解 jq 的基本抽取规则后，即可使用 `JSONLoader` 加载 JSON 文件了。

```python
from langchain_community.document_loaders import JSONLoader
loader = JSONLoader(
    file_path="xxx.json",      # 文件路径
    jq_schema=".",             # jq schema 语法
    text_content=False,        # 抽取的是否是字符串，默认 True
    json_lines=True,           # 是否是 JsonLines 文件（每一行都是 JSON 的文件）
)
```

如下是一个典型的 JsonLines 文件（每行一个 JSON 对象）：
```json
{"name": "周杰轮", "age": 11, "gender": "男"}
{"name": "蔡依临", "age": 12, "gender": "女"}
{"name": "王力鸿", "age": 11, "gender": "男"}
```

> **小结**：
> - `JSONLoader` 依赖 `jq` 库，通过 `pip install jq` 安装
> - `JSONLoader` 使用 `jq` 的解析语法，常见如：`.` 表示根、`[]` 表示数组、`.name` 表示从根取 `name` 的值、`.hobby[1]` 表示取 `hobby` 对应数组的第二个元素、`.[]` 表示将数组内的每个字典（JSON 对象）都取到、`.[].name` 表示取数组内每个字典（JSON）对象的 `name` 对应的值
> - `JSONLoader` 初始化有 4 个主要参数：
>   - `file_path`：文件路径，必填
>   - `jq_schema`：jq 解析语法，必填
>   - `text_content`：抽取到的是否是字符串，默认 `True`，非必填
>   - `json_lines`：是否是 JsonLines 文件，默认 `False`，非必填
> - JsonLines 文件：每一行都是一个独立的字典（Json 对象）

---

### 7.3 PyPDFLoader

LangChain 内支持许多 PDF 的加载器，我们选择其中的 `PyPDFLoader` 使用。`PyPDFLoader` 加载器依赖 PyPDF 库，所以需要安装：`pip install pypdf`

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(
    file_path="",          # 文件路径必填
    mode='page',           # 读取模式，可选 page（按页面划分不同 Document）和 single（单个 Document）
    password='password',   # 文件密码
)
```

---

### 7.4 TextLoader 和文档分割器

#### TextLoader
除了前文学习的三个 Loader 以外，还有一个基本的加载器：`TextLoader`，作用：读取文本文件（如 `.txt`），将全部内容放入一个 `Document` 对象中。

```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader(
    "xxx.txt",
    encoding="utf-8",
)
docs = loader.load()
print(docs)
print(len(docs))    # 结果为 1
```

如果文档很大，加载到一个 `Document` 对象中是否不太合适？需要文档分割器。

#### RecursiveCharacterTextSplitter
`RecursiveCharacterTextSplitter`，递归字符文本分割器，主要用于按自然段落分割大文档。是 LangChain 官方推荐的默认字符分割器。它在保持上下文完整性和控制片段大小之间实现了良好平衡，开箱即用效果佳。

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader(
    "../P3_LangChainRAG开发/data/Python基础语法.txt",
    encoding="utf-8",
)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 分段的最大字符数
    chunk_overlap=50,    # 分段之间允许重叠的字符数
    # 文本分段依据
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    # 字符统计依据（函数）
    length_function=len,
)
split_docs = splitter.split_documents(docs)
```

> **小结**：
> - `TextLoader` 是一个简单的加载器，可以加载文本文件内容，返回仅有一个 `Document` 对象的 list
> - `RecursiveCharacterTextSplitter` 递归字符文本分割器，是 LangChain 官方推荐的默认分割器
> - 基于文本的自然段落分割大文档为小文档
> - 可以指定小文档的最大字符数、重叠字符数
> - 可以手动指定段落划分的依据（符号）以及字符数量统计函数

---

## 八、LangChain 组件：Vector Stores（向量存储）

基于 LangChain 的向量存储，存储嵌入数据，并执行相似性搜索。

这部分开发主要涉及到：
- 如何文本转向量（前文已经学习）
- 创建向量存储，基于向量存储完成：存入向量、删除向量、向量检索

LangChain 为向量存储提供了统一接口：
- `add_documents`
- `delete`
- `similarity_search`

### 8.1 内置向量存储的使用（InMemoryVectorStore）

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings())

# 添加文档到向量存储，并指定 id
vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])

# 删除文档（通过指定的 id 删除）
vector_store.delete(ids=["id1"])

# 相似性搜索
similar_docs = vector_store.similarity_search("your query here", 4)
```

### 8.2 外部向量存储的使用（Chroma）

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
```

> **小结**：
> - LangChain 内提供向量存储功能，可以基于：
>   - `InMemoryVectorStore`，完成内存向量存储
>   - `Chroma`，外部数据库向量存储
> - 向量存储类均提供 3 个通用 API 接口：
>   - `add_document`：添加文档到向量存储
>   - `delete`：从向量存储中删除文档
>   - `similarity_search`：相似度搜索
> - 整体向量存储使用流程如下（RAG 流程）：
>   - 文档加载 → 分割 → 嵌入 → 存入向量库
>   - 用户查询 → 嵌入 → 向量检索 → 构建提示词 → LLM 生成

---

## 九、检索向量并构建提示词

向量存储的实例，通过 `add_texts(list[str])` 方法可以快速添加到向量存储中。

**流程：**
1. 先通过向量存储检索匹配信息
2. 将用户提问和匹配信息一同封装到提示词模板中提问模型

---

## 十、LangChain 组件：RunnablePassthrough

让向量检索加入链？使用 `RunnablePassthrough` 类。（具体用法在后续课程中展开，此处仅作提及）

## 十一、工具tool
### 11.1 @toll

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

### 11.2 BaseTool

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













```




