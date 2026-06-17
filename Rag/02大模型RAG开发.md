# 1. RAG 介绍

## 1.1 理解什么是 RAG

通用的大模型存在一些问题：

- LLM 的知识不是实时的，模型训练好后不具备自动更新知识的能力，会导致部分信息滞后
- LLM 领域知识是缺乏的，大模型的知识来源于训练数据，这些数据主要来自公开的互联网和开源数据集，无法覆盖特定领域或高度专业化的内部知识
- 幻觉问题，LLM 有时会在回答中生成看似合理但实际上是错误的信息
- 数据安全性

RAG（Retrieval-Augmented Generation）即检索增强生成，为大模型提供了从特定数据源检索到的信息，以此来修正和补充生成的答案。可以总结为一个公式：

**RAG = 检索技术 + LLM 提示**

## 1.2 RAG 解决什么问题

- 解决知识实效性问题：大模型的训练数据有截止时间，RAG 可以接入最新文档（如公司财报、政策文件），让模型输出 “与时俱进”
- 降低模型幻觉：模型的回答基于检索到的事实性资料，而非纯靠自身记忆，大幅减少编造信息的概率
- 无需重新训练模型：相比微调（Fine-tuning），RAG 只需更新知识库，成本更低、效率更高

## 1.3 RAG 的工作流程

### 标准流程

RAG 标准流程由索引（Indexing）、检索（Retriever）和生成（Generation）三个核心阶段组成。

**索引阶段**：通过处理多种来源多种格式的文档提取其中文本，将其切分为标准长度的文本块（chunk），并进行嵌入向量化（embedding），向量存储在向量数据库（vector database）中。

- 加载文件 → 内容提取 → 文本分割，形成 chunk → 文本向量化 → 存向量数据库

**检索阶段**：用户输入的查询（query）被转化为向量表示，通过相似度匹配从向量数据库中检索出最相关的文本块。

- query向量化 → 在文本向量中匹配出与问句向量相似的 top_k 个

**生成阶段**：检索到的相关文本与原始查询共同构成提示词（Prompt），输入大语言模型（LLM），生成精确且具备上下文关联的回答。

- 匹配出的文本作为上下文和问题一起添加到 prompt 中 → 提交给 LLM 生成答案

### 离线准备线 / 在线服务线

RAG 工作分为两条线：离线准备线 和 在线服务线。

# 2. 向量的基础概念

## 2.1 向量是什么

向量（Vector）就是文本的 “数学身份证”：它把一段文字的语义信息，转换成一串固定长度的数字列表，让计算机能 “看懂” 文字的含义并做相似度计算。

简单来说，就是让计算机更方便地理解不同的文本内容，是否表述的是一个意思。

示例：
- “如何快速学习 RAG” → [1, 3, 6, 2, 1, 7, 9, 10]
- “RAG 如何快速学会” → [2, 4, 5, 1, 2, 6, 10, 9]
- “什么玩意” → 数字随便填的仅表述概念，非真实向量

## 2.2 向量嵌入与匹配

向量嵌入的过程，我们一般选用合适的文本嵌入模型来完成。在向量匹配的过程中，如何识别两段文本是否表述相似的含义，主要可以通过如余弦相似度等算法来完成。

示例（下列案例中向量为示例，仅描述概念，非真实向量）：

- A：“如何快速学打篮球” → [0.2, 0.5, 0.8]
- B：“打篮球怎么学得快” → [0.18, 0.52, 0.79]
- C：“运动后吃什么好呢” → [0.9, 0.1, 0.2]

通过余弦相似度算法可以计算得到：
- A 和 B 相似度 0.999789
- A 和 C 相似度 0.361446

由此可通过精确的数学计算，去匹配两段文本是否描述同一个意思，提高语义匹配的效率和精度。

文本嵌入模型（如 text-embedding-v1）通过深度学习等技术，从文本提取语义特征并映射为固定长度的数字序列。

## 2.3 向量的维度

如何更为精准地完成语义匹配，生成向量的维度是一个很重要的指标。

如 text-embedding-v1 模型，可以生成 1536 维的向量（一段文本固定得到 1536 个数字序列），比较实用。1536 个数字表示这段文本在 1536 个主题（抽象的语义特征）方向上的得分（强度）。

生成向量的维度越多，就能更好地记录文本的语义特征，做语义匹配会更加精准。但更多的向量会在计算、存储和匹配过程中带来更大的压力。选择合适的向量维度需要在精确和性能之间做平衡。一般 1536 维算是比较好的选择。

**核心概念总结：**
- 向量（Vector）就是文本的 “数学身份证”
- 向量的计算（文本嵌入过程），可借助文本嵌入模型实现，如 text-embedding-v1
- 向量的匹配通过算法实现，如余弦相似度
- 向量的维度表示一段文本在多个抽象语义特征方面的强度
- 维度数代表模型用多少个抽象语义特征来描述文本
- 维度越多，做语义匹配越精准，但性能压力也会增大

# 3. 【扩展】余弦相似度

## 3.1 方向与长度

向量的数字序列，共同决定了向量在高维空间中的方向和长度。而余弦相似度主要就是撇除长度的影响，得到方向的夹角。夹角越小越相似，即方向相同。

一维向量示例：
- 向量 [-0.5]、[0.5]、[1] 的方向和长度

二维向量示例：
- [0.5, 0.5]、[0.7, 0.7]、[0.7, 0.5]、[0.5, -0.5]、[-0.5, -0.5]、[-0.6, 0.5] 的方向和长度

余弦相似度主要匹配的就是：同向（无所谓长度）。我们能直接发现 [0.5, 0.5] 和 [0.7, 0.7] 是同向不同长，那计算机如何判定就依赖余弦相似度算法了。

PS：3维乃至更高维度难以描述，但概念一致。

## 3.2 余弦相似度计算

在文本向量语义匹配中，余弦相似度是衡量两个向量方向相似程度的核心算法，即判断两段文本语义是否相近。

**余弦相似度 = 两个向量的点积 ÷ 两个向量模长的乘积**

以 A[0.5, 0.5]、B[0.7, 0.7]、C[0.7, 0.5]、D[-0.6, -0.5] 为例：

- 点积：两个向量在同维度的乘积之和。向量 AB 点积：vec_a[0]×vec_b[0] + vec_a[1]×vec_b[1] + ... + vec_a[n]×vec_b[n]。如 AB 的点积是：0.5×0.7 + 0.5×0.7 = 0.74
- 模长：单个向量不同维度的平方之和开根号，如 A 的模长是：√(0.5×0.5 + 0.5×0.5)。向量模长：||vec|| = √(vec[0]² + vec[1]² + ... + vec[n]²)。如向量 A 的模长：√(0.5×0.5 + 0.5×0.5)（√ 是开根号）

AB 余弦相似度：(0.5×0.7 + 0.5×0.7) ÷ (√(0.5×0.5 + 0.5×0.5) × √(0.7×0.7 + 0.7×0.7)) = 1.0

AC 余弦相似度：(0.5×0.7 + 0.5×0.5) ÷ (√(0.5×0.5 + 0.5×0.5) × √(0.7×0.7 + 0.5×0.5)) ≈ 0.986

AD 余弦相似度：(0.5×(-0.6) + 0.5×(-0.5)) ÷ (√(0.5×0.5 + 0.5×0.5) × √((-0.6)×(-0.6) + (-0.5)×(-0.5))) ≈ -0.996

# 4. LangChain 简介

LangChain 由 Harrison Chase 创建于 2022 年 10 月，它是围绕 LLMs（大语言模型）建立的一个框架。LangChain 自身并不开发 LLMs，它的核心理念是为各种 LLMs 实现通用的接口，把 LLMs 相关的组件“链接”在一起，简化 LLMs 应用的开发难度，方便开发者快速地开发复杂的 LLMs 应用。

## 4.1 LangChain 主要功能

- Prompts：优化提示词（提示词工程）
- Models：调用各类模型
- History：管理会话历史记录（记忆）
- Indexes：管理和分析各类文档
- Chains：构建功能的执行链条
- Agent：构建智能体

**All in LangChain 一站齐活**

LangChain 是一个开发 LLM 相关业务功能的集大成者，是一个 Python 的第三方库，提供了各种功能的 API。

- 提供提示词优化的相关功能 API
- 调用各类模型的功能 API
- 会话记忆的相关功能 API
- 各类文档管理分析的功能 API
- 构建 Agent 智能体的相关功能 API
- 各类功能链式执行的能力

LangChain 是后续学习 RAG 开发的主力框架。

> Full LLM power—you only need LangChain

# 5. 环境部署

## 5.1 LangChain 安装

```bash
pip install langchain langchain-community langchain-ollama langchain-chroma dashscope chromadb bs4 jq
```

- `langchain`：核心包
- `langchain-community`：社区支持包，提供了更多的第三方模型调用（我们用的阿里云千问模型就需要这个包）
- `langchain-ollama`：Ollama 支持包，支持调用 Ollama 托管部署的本地模型
- `langchain-chroma`：ChromaDB 支持包，支持调用 ChromaDB
- `dashscope`：阿里云通义千问的 Python SDK
- `chromadb`：轻量向量数据库（后续使用）
- `bs4`：BeautifulSoup4 库，协助解析 HTML 文档（后续学习文档加载器使用）

# 6. LangChain 组件：Models

## 6.1 模型类型

LangChain 模型组件提供了与各种模型的集成，并为所有模型提供一个精简的统一接口。LangChain 目前支持三种类型的模型：

- **LLMs（大语言模型）**：是技术范畴的统称，指基于大参数量、海量文本训练的 Transformer 架构模型，核心能力是理解和生成自然语言，主要服务于文本生成场景
- **Chat Models（聊天模型）**：是应用范畴的细分，是专为对话场景优化的 LLMs，核心能力是模拟人类对话的轮次交互，主要服务于聊天场景
- **Embeddings Models（嵌入模型）**：文本嵌入模型接收文本作为输入，得到文本的向量

LangChain 支持的三类模型，它们的使用场景不同，输入和输出不同，开发者需要根据项目需要选择相应。我们所用的阿里云通义千问系列主要来自于 `langchain_community` 包。

## 6.2 LLMs（阿里云大语言模型的访问）

LLMs 使用场景最多，常用大模型的下载库：https://huggingface.co/models 和 https://modelscope.cn/models

以通义千问为例：

```python
from langchain_community.llms.tongyi import Tongyi

# 实例化模型
llm = Tongyi(model='qwen-max')
# 模型推理
res = llm.invoke("帮我讲个笑话吧")
print(res)
```

## 6.3 LLMs（Ollama 本地大语言模型的访问）

如果要访问本地 Ollama 的模型，简单更改一下代码。通过 `langchain_ollama` 包导入 `OllamaLLM` 类即可（请确保 Ollama 已经启动并提前下载好要使用的模型）。

```python
from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")
# 通过 invoke 方法去调用模型
res = model.invoke(input="你是谁呀能做什么？")
print(res)
```

- 通过 `from langchain_community.llm.tongyi import Tongyi` 导入通义千问系列的支持
- 通过 `from langchain_ollama import OllamaLLM` 导入 Ollama 系列的支持
- 创建好模型对象后，通过 `invoke` 对模型发起提问，并可以直接打印输出结果

## 6.4 模型的流式输出

如果需要流式输出结果，需要将模型的 `invoke` 方法改为 `stream` 方法即可。

- `invoke` 方法：一次性返回完整结果
- `stream` 方法：逐段返回结果，流式输出

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

模型对象有 2 个方法去调用模型：
- `invoke`：调用模型，一次性返回完整结果
- `stream`：调用模型，逐段流式输出

这两个方法是新版 LangChain（1.0 版本后）中基于 Runnable 接口的通用核心方法。绝大多数组件（如提示词模板、链、向量检索、工具调用等）都支持这两个方法，这也是 LangChain 设计的核心统一范式。

## 6.5 Chat Models（聊天模型）

聊天消息包含下面几种类型，使用时需要按照约定传入合适的值：

- `AIMessage`：就是 AI 输出的消息，可以是针对问题的回答（OpenAI 库中的 assistant 角色）
- `HumanMessage`：人类消息就是用户信息，由人给出的信息发送给 LLMs 的提示信息，比如“实现一个快速排序方法”（OpenAI 库中的 user 角色）
- `SystemMessage`：可以用于指定模型具体所处的环境和背景，如角色扮演等。你可以在这里给出具体的指示，比如“作为一个代码专家”，或者“返回 json 格式”（OpenAI 库中的 system 角色）

### 6.5.1 单独使用 HumanMessage

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

### 6.5.2 使用 SystemMessage + HumanMessage

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

### 6.5.3 使用 SystemMessage + HumanMessage + AIMessage

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

## 6.6 消息的简写形式

`SystemMessage`、`HumanMessage`、`AIMessage` 可以有如下的简写形式：

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

区别和优势在于：
- 使用类对象的方式：是静态的，一步到位就得到了 Message 类的类对象
- 简写形式：是动态的，需要在运行时由 LangChain 内部机制转换为 Message 类对象。由于是动态，需要转换步骤，所以简写形式支持内部填充 {变量} 占位，可在运行时填充具体值（后续学习提示词模板时用到）

```python
messages = [
    ("system", "今天的天气是 {weather}"),
    ("human", "我的名字是：{name}"),
    ("ai", "欢迎 {lastname} 先生"),
]
```

## 6.7 Embeddings Models（文本嵌入模型）

嵌入模型的特点：将字符串作为输入，返回一个浮点数的列表（向量）。在 NLP 中，Embedding 的作用就是将数据进行文本向量化。

**阿里云千问模型访问方式：**
```python
from langchain_community.embeddings import DashScopeEmbeddings

# 初始化嵌入模型对象，其默认使用模型是：text-embedding-v1
embed = DashScopeEmbeddings()
# 测试
print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(['我喜欢你', '我稀饭你', '晚上吃啥']))
```

**本地 Ollama 模型访问方式：**
通过 `langchain_ollama` 导入 `OllamaEmbeddings` 使用，其余不变。
```python
from langchain_ollama import OllamaEmbeddings

# 初始化嵌入模型对象，其默认使用模型是：qwen3-embedding
embed = OllamaEmbeddings(model="qwen3-embedding")
# 测试
print(embed.embed_query("我喜欢你"))
print(embed.embed_documents(['我喜欢你', '我稀饭你', '晚上吃啥']))
```

## 6.8 模型使用总结

目前所掌握的 LangChain API 如下：
- LangChain 在模型的支持上主要基于 `langchain_community` 包提供。
- 主要支持三类模型：
  - LLMs：大语言模型，主用于文本生成
  - Chat Model：聊天模型，主用于多轮次对话的聊天场景
  - Embeddings Model：文本嵌入模型，主用于生成文本向量
- LangChain 框架和 OpenAI 库一样，提供三种角色：
  - `HumanMessage` 类，即 User 角色
  - `AIMessage` 类，即 Assistant 角色
  - `SystemMessage` 类，即 System 角色

# 7. LangChain 组件：Prompts

## 7.1 通用 Prompt（zero-shot）

提示词优化在模型应用中非常重要，LangChain 提供了 `PromptTemplate` 类，用来协助优化提示词。`PromptTemplate` 表示提示词模板，可以构建一个自定义的基础提示词模板，支持变量的注入，最终生成所需的提示词。

**标准写法：**
```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 帮忙起名字，请简略回答。"
)
# 变量注入，生成提示词文本
prompt_text = prompt_template.format(lastname="张", gender="女儿")

model = Tongyi(model="qwen-max")        # 创建模型对象
res = model.invoke(input=prompt_text)   # 调用模型获取结果
print(res)
```

**基于 chain 链的写法：**
```python
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.tongyi import Tongyi

prompt_template = PromptTemplate.from_template(
    "我的邻居姓{lastname}, 刚生了{gender}, 帮忙起名字，请简略回答。"
)
model = Tongyi(model="qwen-max")        # 创建模型对象
# 生成链
chain = prompt_template | model
# 基于链，调用模型获取结果
res = chain.invoke(input={"lastname": "曹", "gender": "女儿"})
print(res)
```

基于 `PromptTemplate` 类可以得到提示词模板，支持基于模板注入变量得到最终提示词。zero-shot 思想下，可以基于 `PromptTemplate` 直接完成。few-shot 思想下，需要更换为 `FewShotPromptTemplate`（后续学习）。

PS：使用 `PromptTemplate` 还不如自己手动拼接字符串？
- 使用 Template 模板构建提示词，在大型工程中更容易做标准化模板
- Template 模板类，支持 LangChain 框架的链式调用（Runnable 接口）
- `PromptTemplate`
- `FewShotPromptTemplate`（后续学习）
- `ChatPromptTemplate`（后续学习）

## 7.2 FewShotPromptTemplate

```python
from langchain_core.prompts import FewShotPromptTemplate

FewShotPromptTemplate(
    examples=None,
    example_prompt=None,
    prefix=None,
    suffix=None,
    input_variables=None
)
```

参数：
- `examples`：示例数据，list，内套字典
- `example_prompt`：示例数据的提示词模板
- `prefix`：组装提示词，示例数据前内容
- `suffix`：组装提示词，示例数据后内容
- `input_variables`：列表，注入的变量列表

**组装 FewShotPromptTemplate 对象并获得最终提示词：**
```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

example_template = PromptTemplate.from_template("单词:{word}, 反义词:{antonym}")
example_data = [                            # 示例数据，list 内套字典
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]
few_shot_prompt = FewShotPromptTemplate(   # FewShot 提示词模板对象
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

example_template = PromptTemplate(         # 示例的提示词模板对象
    input_variables=['word', 'antonym'],
    template="word: {word}, antonym: {antonym}"
)
example_data = [                           # 示例数据，list 内套字典
    {"word": "大", "antonym": "小"},
    {"word": "上", "antonym": "下"}
]
few_shot_prompt = FewShotPromptTemplate(   # FewShot 提示词模板对象
    examples=example_data,
    example_prompt=example_template,
    prefix="给出给定词的反义词，有如下示例：\n",
    suffix="基于示例告诉我：{input_word}的反义词是？",
    input_variables=['input_word']
)
# 获得最终提示词
prompt_text = few_shot_prompt \
    .invoke(input={"input_word": "左"}) \
    .to_string()

model = ChatTongyi(model="qwen3-max")
for chunk in model.stream(input=prompt_text):
    print(chunk.content, end="", flush=True)
```

`FewShotPromptTemplate` 类对象构建需要 5 个核心参数：
- `example_prompt`：示例数据的提示词模板
- `examples`：示例数据，list，内套字典
- `prefix`：组装提示词，示例数据前内容
- `suffix`：组装提示词，示例数据后内容
- `input_variables`：列表，注入的变量列表

## 7.3 模板类的 format 和 invoke 方法

在 `PromptTemplate`（通用提示词模板）和 `FewShotPromptTemplate`（FewShot 提示词模板）的使用中，我们使用了如下：

- 模板对象的 `format` 方法：`format(**kwargs)` → 字符串
- 模板对象的 `invoke` 方法：`invoke(input: dict)` → `PromptValue`

`format` 和 `invoke` 的区别在于：
- `format` 直接返回字符串
- `invoke` 返回 `PromptValue` 对象，可进一步调用 `.to_string()` 或 `.to_messages()`

## 7.4 ChatPromptTemplate

- `PromptTemplate`：通用提示词模板，支持动态注入信息。
- `FewShotPromptTemplate`：支持基于模板注入任意数量的示例信息。
- `ChatPromptTemplate`：支持注入任意数量的历史会话信息。

通过 `from_messages` 方法，从列表中获取多轮次会话作为聊天的基础模板。PS：前面 `PromptTemplate` 类用的 `from_template` 仅能接入一条消息，而 `from_messages` 可以接入一个 list 的消息。

```python
from langchain_core.prompts import ChatPromptTemplate

ChatPromptTemplate.from_messages(
    [
        ("system", "........"),
        ("ai", "........"),
        ......,
        ("human", "........")
    ]
)
```

历史会话信息并不是静态的（固定的），而是随着对话的进行不停地积攒，即动态的。所以，历史会话信息需要支持动态注入。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "........"),
        ("ai", "........"),
        MessagesPlaceholder("history"),   # 作为占位，提供 history 作为占位的 key
        ("human", "........")
    ]
)

history_data = [
    ("human", "..."),
    ("ai", "..."),
    ("human", "..."),
    ("ai", "...")
]
# 基于 invoke 动态注入历史会话记录，必须是 invoke，format 无法注入
prompt_value = chat_template.invoke({"history": history_data})
```

ChatPromptTemplate 也可以基于聊天模型，并组装聊天历史的模式，做提示词工程。

**few-shot 提示方式：求反义词**

```python
examples = [
    {"word": "开心", "antonym": "难过"},
    {"word": "高", "antonym": "矮"},
]

from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "给出每个单词的反义词"),
        # 存储多轮对话的历史记录，history 是占位符名称，后续从字典按 history 作为 key 取 value 替代内容
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

model = ChatTongyi(model="qwen3-max")
# StrOutputParser() LangChain 内置的结果解析器，可以直接提取结果文本内容，剔除其余元数据信息
chain = prompt | model | StrOutputParser()

# 无历史会话的提问
for chunk in chain.stream(input={"history": [], "question": "粗"}):
    print(chunk)

print("*" * 20)

# 带有历史的提问，用 HumanMessage（用户消息）和 AIMessage（模型消息）封装历史对话
history = [     # history 要求是一个列表，内部封装用户和 AI 的对话记录
    HumanMessage(content="开心"),   # 对应 ("human", "开心")
    AIMessage(content="难过"),      # 对应 ("ai", "难过")
    HumanMessage(content="高"),     # 对应 ("human", "高")
    AIMessage(content="矮")         # 对应 ("ai", "矮")
]
# 简化写法，元组的第一个元素是角色（标准角色名 human/ai），第二个元素是消息
# history = [
#     ("human", "开心"), ("ai", "难过"),
#     ("human", "高"), ("ai", "矮")
# ]

for chunk in chain.stream(input={"history": history, "question": "粗"}):
    print(chunk)
```

# 8. LangChain 组件：Chains

## 8.1 链的基础使用

「将组件串联，上一个组件的输出作为下一个组件的输入」是 LangChain 链（尤其是 `|` 管道链）的核心工作原理，这也是链式调用的核心价值：实现数据的自动化流转与组件的协同工作。

```python
chain = prompt_template | model
```

核心前提：即 Runnable 子类对象才能入链（以及 Callable、Mapping 接口子类对象也可加入）。我们目前所学习到的组件，均是 Runnable 接口的子类。

通过 `|` 链接提示词模板对象和模型对象，返回值 chain 对象是 `RunnableSerializable` 对象，是 Runnable 接口的直接子类，也是绝大多数组件的父类。通过 `invoke` 或 `stream` 进行阻塞执行或流式执行。

组成的链在执行上有：上一个组件的输出作为下一个组件的输入的特性。

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

LangChain 中链是一种将各个组件串联在一起，按顺序执行，前一个组件的输出作为下一个组件的输入。可以通过 `|` 符号来让各个组件形成链，成链的各个组件需是 Runnable 接口的子类。形成的链是 `RunnableSerializable` 对象（Runnable 接口子类），可通过链调用 `invoke` 或 `stream` 触发整个链条的执行。

## 8.2 `|` 运算符重载

前文代码中：`chain = chat_prompt_template | model` 在语法上使用了 `|` 运算符的重写。

在 Python 中，运算符（如 `+`、`|`）的行为由类的魔法方法决定。例如：
- `a + b` 本质调用的是 `a.__add__(b)`
- `a | b` 本质调用的是 `a.__or__(b)`

只需要自行实现类的 `__or__` 方法，即可对 `|` 符号的功能进行重写。

示例：让 `a|b|c` 的代码得到一个自定义的类对象（类似列表即 [a, b, c]），调用 run 方法依次输出 a、b、c。我们需要重写 `|` 即 `__or__` 方法。

```python
class Test(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"Test({self.name})"
    def __or__(self, other):
        return MySequence(self, other)

class MySequence(object):
    def __init__(self, *args):
        self.sequence = []
        for arg in args:
            self.sequence.append(arg)
    def __or__(self, other):
        self.sequence.append(other)
        return self
    def run(self):
        for arg in self.sequence:
            print(arg)

if __name__ == "__main__":
    a = Test("a")
    b = Test("b")
    c = Test("c")
    d = a | b | c
    d.run()
    print(type(d))
```

# 9. LangChain 组件：输出解析器

## 9.1 StrOutputParser

有如下代码，想要以第一次模型的输出结果，第二次去询问模型：

```python
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi

model = ChatTongyi(model="qwen3-max")
prompt = PromptTemplate.from_template(
    "我邻居姓：{lastname}, 刚生了{gender}，请起名，仅告知名字无需其它内容"
)
chain = prompt | model | model
res = chain.invoke({"lastname": "张", "gender": "女儿"})
print(res.content)
```

运行报错（ValueError: Invalid input type <class 'langchain_core.messages.ai.AIMessage'>. Must be a PromptValue, str, or list of BaseMessages.）

错误的主要原因是：prompt 的结果是 `PromptValue` 类型，输入给了 model，model 的输出结果是 `AIMessage`。模型（ChatTongyi）源码中关于 `invoke` 方法明确指定了 input 的类型：

```python
@override
def invoke(
    self,
    input: LanguageModelInput,
    config: RunnableConfig | None = None,
    *,
    stop: list[str] | None = None,
    **kwargs: Any,
) -> AIMessage:

LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
"""Input to a language model."""
```

需要做类型转换，可以借助 LangChain 内置的解析器 `StrOutputParser`。

`StrOutputParser` 是 LangChain 内置的简单字符串解析器，可以将 `AIMessage` 解析为简单的字符串，符合了模型 `invoke` 方法要求（可传入字符串，不接收 `AIMessage` 类型）。它是 Runnable 接口的子类，可以加入链。

```python
parser = StrOutputParser()
chain = prompt | model | parser | model
```

`StrOutputParser` 是 LangChain 内置的简单字符串解析器，可以将 `AIMessage` 类型转换为基础字符串，可以加入 chain 作为组件存在（Runnable 接口子类）。

## 9.2 JsonOutputParser 与多模型执行链

在前面我们完成了这样的需求去构建多模型链，不过这种做法并不标准，因为上一个模型的输出没有被处理就输入下一个模型。正常情况下我们应该有如下处理逻辑：上一个模型的输出结果，应该作为提示词模板的输入，构建下一个提示词，用来二次调用模型。

```python
chain = prompt | model | parser | model | parser
```

`invoke`｜`stream` 初始输入 → 提示词模板 → 模型 → 数据处理 → 提示词模板 → 模型 → 解析器 → 结果

根据输出和输入的要求：
- 模型的输出为：`AIMessage` 类对象
- 提示词模板要求输入如右侧代码：

```python
def invoke(self, input: dict, config: RunnableConfig | None = None, **kwargs: Any) -> PromptValue:
```

所以，我们需要完成：将模型输出的 `AIMessage` → 转为字典 → 注入第二个提示词模板中，形成新的提示词（PromptValue 对象）。

`StrOutputParser` 不满足（AIMessage → Str），更换 `JsonOutputParser`（AIMessage → Dict(JSON)）。

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

在构建链的时候要注意整体兼容性，注意前后组件的输入和输出要求：
- 模型输入：PromptValue 或字符串或序列（BaseMessage、list、tuple、str、dict）
- 模型输出：AIMessage
- 提示词模板输入：要求是字典
- 提示词模板输出：PromptValue 对象
- StrOutputParser：AIMessage 输入、str 输出
- JsonOutputParser：AIMessage 输入、dict 输出

```
first_prompt | model | json_parser | second_prompt | model | str_parser
  PromptValue   AIMessage    字典        PromptValue    AIMessage    字典
                                                                     字符串
```

## 9.3 RunnableLambda 与函数加入链

前文我们根据 `JsonOutputParser` 完成了多模型执行链条的构建。除了 `JsonOutputParser` 这类固定功能的解析器之外，我们也可以自己编写 Lambda 匿名函数来完成自定义逻辑的数据转换，想怎么转换就怎么转换，更自由。

想要完成这个功能，可以基于 `RunnableLambda` 类实现。`RunnableLambda` 类是 LangChain 内置的，将普通函数等转换为 Runnable 接口实例，方便自定义函数加入 chain。

语法：`RunnableLambda(函数对象或 lambda 匿名函数)`

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

**函数直接入链：**
```python
chain = first_prompt | model | (lambda ai_msg: {"name": ai_msg.content}) | second_prompt | model | str_parser
```

跳过 `RunnableLambda` 类，直接让函数加入链也是可以的。因为 Runnable 接口类在实现 `__or__` 的时候，支持 Callable 接口的实例。函数就是 Callable 接口的实例。其本质是将函数自动转换为 `RunnableLambda`。

```python
def __or__(
    self,
    other: Runnable[Any, Other]
    | Callable[[Iterator[Any]], Iterator[Other]]
    | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
    | Callable[[Any], Other]
    | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
) -> RunnableSerializable[Input, Other]:
```

如果要在链中加入自定义函数，可以选择：
- 将函数封装入 `RunnableLambda` 类对象，其是 Runnable 接口实例，可以直接入链
- 直接将函数入链，函数会自动转换为 `RunnableLambda` 对象

# 10. LangChain 组件：Memory

## 10.1 临时会话记忆（InMemoryChatMessageHistory）

如果想要封装历史记录，除了自行维护历史消息外，也可以借助 LangChain 内置的历史记录附加功能。

LangChain 提供了 History 功能，帮助模型在有历史记忆的情况下回答。
- 基于 `RunnableWithMessageHistory` 在原有链的基础上创建带有历史记录功能的新链（新 Runnable 实例）
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

获取指定会话 ID 的历史会话记录函数：
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

`RunnableWithMessageHistory` 是 LangChain 内 Runnable 接口的实现，主要用于创建一个带有历史记忆功能的 Runnable 实例（链）。它在创建的时候需要提供一个 `BaseChatMessageHistory` 的具体实现（用来存储历史消息），`InMemoryChatMessageHistory` 可以实现在内存中存储历史。

额外的，如果想要在 `invoke` 或 `stream` 执行链的同时，将提示词 print 出来，可以在链中加入自定义函数实现。注意：函数的输入应原封不动返回出去，避免破坏原有业务，仅在 return 之前，print 所需信息即可。

## 10.2 长期会话记忆（自实现 FileChatMessageHistory）

使用 `InMemoryChatMessageHistory` 仅可以在内存中临时存储会话记忆，一旦程序退出，则记忆丢失。

`InMemoryChatMessageHistory` 类继承自 `BaseChatMessageHistory`，在官方注释中给出了相关实现的指南，并给出了基于文件的历史消息存储示例代码。我们可以自行实现一个基于 Json 格式和本地文件的会话数据保存。

**FileChatMessageHistory 类实现核心思路：**
- 基于文件存储会话记录，以 session_id 为文件名，不同 session_id 有不同文件存储消息
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

# 11. LangChain 组件：Document Loaders（文档加载器）

文档加载器提供了一套标准接口，用于将不同来源（如 CSV、PDF 或 JSON 等）的数据读取为 LangChain 的文档格式。这确保了无论数据来源如何，都能对其进行一致性处理。

文档加载器（内置或自行实现）需实现 `BaseLoader` 接口。

`Class Document` 是 LangChain 内文档的统一载体，所有文档加载器最终返回此类的实例。一个基础的 `Document` 类实例创建如下：

```python
from langchain_core.documents import Document
document = Document(
    page_content="Hello, world!", metadata={"source": "https://example.com"}
)
```

`Document` 类其核心记录了：
- `page_content`：文档内容
- `metadata`：文档元数据（字典）

不同的文档加载器可能定义了不同的参数，但是其都实现了统一的接口（方法）：
- `load()`：一次性加载全部文档
- `lazy_load()`：延迟流式传输文档，对大型数据集很有用，避免内存溢出

一个简单的 CSVLoader 的使用示例如下：

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(
    ...  # 初始化参数
)
# 一次性加载全部文档
documents = loader.load()
# 对于大数据集，分段返回文档
for document in loader.lazy_load():
    print(document)
```

LangChain 内置了许多文档加载器，详细参见官方文档：https://docs.langchain.com/oss/python/integrations/document_loaders

我们简单学习如下几个常用的文档加载器：CSVLoader、JSONLoader、PDFLoader。

## 11.1 CSVLoader

```python
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./xxx.csv")
data = loader.load()
print(data)
```

CSVLoader 自定义 CSV 文件的解析和加载：
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

- LangChain 内置了许多种类的文档加载器
- 文档加载器均继承于 `BaseLoader` 类
- 返回 `Document` 类型的对象
- `load` 方法一次性批量加载（返回 list 内含 Document 对象），如内容过多可能 list 太大，出现内存溢出问题
- `lazy_load` 方法会得到生成器对象，可用 for 循环依次获取单个 Document 对象，适用于大文档避免内存存不下
- CSVLoader 用于加载 CSV 文件，加载成功得到的即 Document 对象

## 11.2 JSONLoader

JSONLoader 用于将 JSON 数据加载为 Document 类型对象。使用 JSONLoader 需要额外安装：`pip install jq`。jq 是一个跨平台的 json 解析工具，LangChain 底层对 JSON 的解析就是基于 jq 工具实现的。

将 JSON 数据的信息抽取出来，封装为 Document 对象，抽取的时候依赖 `jq_schema` 语法。

常见 jq 语法：
- `.` 表示整个 JSON 对象（根）
- `[]` 表示数组
- `.name` 表示抽取 name 的值
- `.hobby` 表示抽取爱好数组
- `.hobby[1]` 或 `.hobby.[1]` 表示抽取第二个爱好
- `.other.addr` 表示抽取地址

示例 JSON：
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

数组示例：
```json
[
    {"name": "周杰轮", "age": 11, "gender": "男"},
    {"name": "蔡依临", "age": 12, "gender": "女"},
    {"name": "王力鸿", "age": 11, "gender": "男"}
]
```
- `.[]` 得到 3 个字典
- `.[].name` 表示抽取全部的 name，即得到 3 个 name 信息

了解 jq 的基本抽取规则后，即可使用 JSONLoader 加载 JSON 文件了。

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
```
{"name": "周杰轮", "age": 11, "gender": "男"}
{"name": "蔡依临", "age": 12, "gender": "女"}
{"name": "王力鸿", "age": 11, "gender": "男"}
```

- JSONLoader 依赖 jq 库，通过 `pip install jq` 安装
- JSONLoader 使用 jq 的解析语法，常见如：`.` 表示根、`[]` 表示数组、`.name` 表示从根取 name 的值、`.hobby[1]` 表示取 hobby 对应数组的第二个元素、`.[]` 表示将数组内的每个字典（JSON 对象）都取到、`.[].name` 表示取数组内每个字典（JSON）对象的 name 对应的值
- JSONLoader 初始化有 4 个主要参数：
  - `file_path`：文件路径，必填
  - `jq_schema`：jq 解析语法，必填
  - `text_content`：抽取到的是否是字符串，默认 True，非必填
  - `json_lines`：是否是 JsonLines 文件，默认 False，非必填
- JsonLines 文件：每一行都是一个独立的字典（Json 对象）

## 11.3 PyPDFLoader

LangChain 内支持许多 PDF 的加载器，我们选择其中的 PyPDFLoader 使用。PyPDFLoader 加载器依赖 PyPDF 库，所以需要安装：`pip install pypdf`

```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(
    file_path="",          # 文件路径必填
    mode='page',           # 读取模式，可选 page（按页面划分不同 Document）和 single（单个 Document）
    password='password',   # 文件密码
)
```

## 11.4 TextLoader 和文档分割器

### TextLoader

除了前文学习的三个 Loader 以外，还有一个基本的加载器：TextLoader，作用：读取文本文件（如 .txt），将全部内容放入一个 Document 对象中。

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

如果文档很大，加载到一个 Document 对象中是否不太合适？需要文档分割器。

### RecursiveCharacterTextSplitter

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

- TextLoader 是一个简单的加载器，可以加载文本文件内容，返回仅有一个 Document 对象的 list
- RecursiveCharacterTextSplitter 递归字符文本分割器，是 LangChain 官方推荐的默认分割器
- 基于文本的自然段落分割大文档为小文档
- 可以指定小文档的最大字符数、重叠字符数
- 可以手动指定段落划分的依据（符号）以及字符数量统计函数

# 12. LangChain 组件：Vector Stores（向量存储）

基于 LangChain 的向量存储，存储嵌入数据，并执行相似性搜索。

这部分开发主要涉及到：
- 如何文本转向量（前文已经学习）
- 创建向量存储，基于向量存储完成：存入向量、删除向量、向量检索

LangChain 为向量存储提供了统一接口：
- `add_documents`
- `delete`
- `similarity_search`

## 12.1 内置向量存储的使用（InMemoryVectorStore）

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

## 12.2 外部向量存储的使用（Chroma）

```python
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=DashScopeEmbeddings(),
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
```

LangChain 内提供向量存储功能，可以基于：
- `InMemoryVectorStore`，完成内存向量存储
- `Chroma`，外部数据库向量存储

向量存储类均提供 3 个通用 API 接口：
- `add_document`：添加文档到向量存储
- `delete`：从向量存储中删除文档
- `similarity_search`：相似度搜索

整体向量存储使用流程如下（RAG 流程）：
1. 文档加载 → 分割 → 嵌入 → 存入向量库
2. 用户查询 → 嵌入 → 向量检索 → 构建提示词 → LLM 生成

# 13. 检索向量并构建提示词

向量存储的实例，通过 `add_texts(list[str])` 方法可以快速添加到向量存储中。

流程：
1. 先通过向量存储检索匹配信息
2. 将用户提问和匹配信息一同封装到提示词模板中提问模型

# 14. LangChain 组件：RunnablePassthrough

让向量检索加入链？使用 `RunnablePassthrough` 类。（具体用法在后续课程中展开，此处仅作提及）
