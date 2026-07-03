
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

### 6.1 临时会话记忆（InMemoryChatMessageHistory）

如果想要封装历史记录，除了自行维护历史消息外，也可以借助 LangChain 内置的历史记录附加功能。

LangChain 提供了 History 功能，帮助模型在有历史记忆的情况下回答。

- 基于 `RunnableWithMessageHistory` 在原有链的基础上创建带有历史记录功能的新链（新 `Runnable` 实例）
- 基于 `InMemoryChatMessageHistory` 为历史记录提供内存存储（临时用）

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

# 通过 RunnableWithMessageHistory 获取一个新的带有历史记录功能的 chain
conversation_chain = RunnableWithMessageHistory(
    some_chain,                  # 被附加历史消息的 Runnable，通常是 chain
    None,                        # 获取指定会话 ID 的历史会话的函数
    input_messages_key="input",  # 声明用户输入消息在模板中的占位符
    history_messages_key="chat_history"  # 声明历史消息在模板中的占位符
)
```

**获取指定会话 ID 的历史会话记录函数：**

```python
chat_history_store = {}     # 存放多个会话 ID 所对应的历史会话记录

# 函数传入为会话 ID（字符串类型）
# 函数要求返回 BaseChatMessageHistory 的子类
# BaseChatMessageHistory 类专用于存放某个会话的历史记录
# InMemoryChatMessageHistory 是官方自带的基于内存存放历史记录的类
def get_history(session_id):
    if session_id not in chat_history_store:
        # 返回一个新的实例
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]
```

**完整代码：**

```python
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

def print_prompt(full_prompt):
    print("="*20, full_prompt.to_string(), "="*20)
    return full_prompt

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "你需要根据对话历史回应用户问题。对话历史：{chat_history}。用户当前输入：{input}，请给出回应"
)
base_chain = prompt | print_prompt | model | StrOutputParser()

chat_history_store = {}     # 存放多个会话 ID 所对应的历史会话记录

def get_history(session_id):
    if session_id not in chat_history_store:
        # 存入新的实例
        chat_history_store[session_id] = InMemoryChatMessageHistory()
    return chat_history_store[session_id]

# 通过 RunnableWithMessageHistory 获取一个新的带有历史记录功能的 chain
conversation_chain = RunnableWithMessageHistory(
    base_chain,              # 被附加历史消息的 Runnable，通常是 chain
    get_history,             # 获取历史会话的函数
    input_messages_key="input",          # 声明用户输入消息在模板中的占位符
    history_messages_key="chat_history"  # 声明历史消息在模板中的占位符
)

if __name__ == '__main__':
    # 如下固定格式，配置当前会话的 ID
    session_config = {"configurable": {"session_id": "user_001"}}
    print(conversation_chain.invoke({"input": "小明有一只猫"}, session_config))
    print(conversation_chain.invoke({"input": "小刚有两只狗"}, session_config))
    print(conversation_chain.invoke({"input": "共有几只宠物？"}, session_config))
```

> **小结**：`RunnableWithMessageHistory` 是 LangChain 内 `Runnable` 接口的实现，主要用于创建一个带有历史记忆功能的 `Runnable` 实例（链）。它在创建的时候需要提供一个 `BaseChatMessageHistory` 的具体实现（用来存储历史消息），`InMemoryChatMessageHistory` 可以实现在内存中存储历史。  
> 额外的，如果想要在 `invoke` 或 `stream` 执行链的同时，将提示词 `print` 出来，可以在链中加入自定义函数实现。注意：函数的输入应原封不动返回出去，避免破坏原有业务，仅在 `return` 之前，`print` 所需信息即可。

---

### 6.2 长期会话记忆（自实现 FileChatMessageHistory）

使用 `InMemoryChatMessageHistory` 仅可以在内存中临时存储会话记忆，一旦程序退出，则记忆丢失。

`InMemoryChatMessageHistory` 类继承自 `BaseChatMessageHistory`，在官方注释中给出了相关实现的指南，并给出了基于文件的历史消息存储示例代码。我们可以自行实现一个基于 Json 格式和本地文件的会话数据保存。

**FileChatMessageHistory 类实现核心思路：**

- 基于文件存储会话记录，以 `session_id` 为文件名，不同 `session_id` 有不同文件存储消息
- 继承 `BaseChatMessageHistory` 实现如下 3 个方法：
  - `add_messages`：同步模式，添加消息
  - `messages`：同步模式，获取消息
  - `clear`：同步模式，清除消息

官方在 `BaseChatMessageHistory` 类的注释中提供了一个基于文件存储的示例代码。

**其余核心代码：**

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import messages_from_dict, message_to_dict
from langchain_community.chat_models.tongyi import ChatTongyi
from typing import Sequence, List
import json

# -------------------------- 核心业务逻辑 -------------------------- #
# 1. 初始化通义千问模型
llm = ChatTongyi(model="qwen3-max")

# 2. 定义提示词模板（包含对话历史和用户输入）
prompt = PromptTemplate.from_template("""
你是一个贴心的助手，需要根据对话历史回应用户的问题。
对话历史：
{chat_history}
用户当前输入：
{input}
你的回应：
""")

# 3. 构建基础链（提示词 -> 模型 -> 输出解析）
base_chain = prompt | llm | StrOutputParser()

# 4. 定义会话历史获取函数（为每个会话创建独立的文件存储）
def get_message_history(session_id: str) -> BaseChatMessageHistory:
    """
    根据会话 ID 获取对应的对话历史存储实例
    :param session_id: 会话唯一 ID
    :return: FileChatMessageHistory 实例
    """
    return FileChatMessageHistory(session_id=session_id, storage_path="./chat_history")

# 5. 包装带对话历史的链式调用
conversation_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_message_history,   # 历史获取函数
    input_messages_key="input",   # 用户输入的参数名
    history_messages_key="chat_history",   # 对话历史的参数名
)

# -------------------------- 测试多轮对话 -------------------------- #
if __name__ == "__main__":
    # 会话配置（指定会话 ID，用于区分不同用户/会话）
    session_config = {"configurable": {"session_id": "user_001"}}

    # 第一轮对话
    response1 = conversation_chain.invoke({"input": "小明有1只猫"}, config=session_config)
    print("第一轮：", response1)

    # 第二轮对话
    response2 = conversation_chain.invoke({"input": "小刚有2只狗"}, config=session_config)
    print("\n第二轮：", response2)

    # 第三轮对话（依赖历史上下文）
    response3 = conversation_chain.invoke(
        {"input": "小明和小刚一共有几只宠物?"},
        config=session_config
    )
    print("\n第三轮：", response3)

    # 测试程序重启后读取历史（注释上面的代码，单独运行下面的代码仍能获取历史）
    # response4 = conversation_chain.invoke(
    #     {"input": "分别是什么宠物？"},
    #     config=session_config
    # )
    # print("\n重启后第四轮：", response4)
```

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

**⚙️ 其他重要参数**

`@tool` 装饰器还提供了其他参数，用于控制工具的行为。

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `name` | `str` | 函数名 | 自定义工具名称。 |
| `description` | `str` | 文档字符串 | 自定义工具描述。 |
| `args_schema` | `BaseModel` | `None` | 使用 Pydantic 模型定义更严格的输入模式。 |
| `return_direct` | `bool` | `False` | 如果设为 `True`，工具的输出会直接返回给用户，而不会再次传递给模型进行后续处理。 |

##### 🚀 在 Agent 中使用

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














```




