## 一、PyMilvus 核心操作（基础用法）

### 1. 环境准备与连接

**安装 PyMilvus**

```bash
pip install pymilvus
```

**连接 Milvus 服务**

使用 `MilvusClient` 作为与 Milvus 交互的主要接口。

```python
from pymilvus import MilvusClient

# Milvus Lite 本地文件模式（适合原型开发）
client = MilvusClient("./milvus_demo.db")

# 服务器模式（连接本地 Milvus 服务）
client = MilvusClient(uri="http://localhost:19530")

# 带认证的连接
client = MilvusClient(
    uri="https://localhost:19530",
    token="root:Milvus"
)
```

检查连接状态：
```python
print(client.get_server_version())
```

---

### 2. 数据库管理

```python
from pymilvus import connections, db, utility

# 使用 ORM 方式连接
connections.connect("default", host="localhost", port="19530")

# 创建数据库
db.create_database("my_database")

# 切换数据库
db.using_database("my_database")

# 列出所有数据库
db.list_database()

# 删除数据库
db.drop_database("my_database")
```

---

### 3. Collection（集合）管理

Collection 类似于关系型数据库中的表，是组织和管理向量数据及标量元数据的核心单元。

#### 3.1 使用 MilvusClient 快速创建（最简单方式）

只需指定集合名称和向量维度：

```python
client.create_collection(
    collection_name="quick_setup",
    dimension=128,           # 向量维度
    metric_type="IP"         # 相似度度量类型：IP（内积）或 L2（欧氏距离）
)
```

#### 3.2 使用 Schema 精细定义

```python
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="price", dtype=DataType.FLOAT)
]

# 创建 Schema
schema = CollectionSchema(fields, description="商品集合")

# 创建 Collection
collection = Collection(name="products", schema=schema)
```

**FieldSchema 参数说明**：
- `name`：字段名称
- `dtype`：数据类型（INT64、VARCHAR、FLOAT_VECTOR、BINARY_VECTOR 等）
- `is_primary`：是否为主键
- `max_length`：VARCHAR 类型的最大长度
- `dim`：向量维度

#### 3.3 集合的其他操作

```python
# 列出所有集合
collections = utility.list_collections()

# 加载集合到内存（搜索前必须执行）
collection.load()

# 释放集合（释放内存）
collection.release()

# 删除集合
collection.drop()

# 获取集合信息
print(collection.schema)
print(collection.num_entities)  # 实体数量
```

---

### 4. 数据插入

#### 4.1 使用 MilvusClient 插入

```python
# 准备数据（字典列表形式）
data = [
    {"id": 1, "title": "The Great Gatsby", "embedding": [0.1, 0.2, ...], "price": 29.99},
    {"id": 2, "title": "1984", "embedding": [0.3, -0.1, ...], "price": 19.99}
]

# 插入数据
res = client.insert(
    collection_name="products",
    data=data
)
print(res["insert_count"])  # 插入数量
```

#### 4.2 使用 ORM Collection 插入

**列式组织（按字段）**：
```python
import numpy as np

ids = [101, 102, 103]
titles = ["The Great Gatsby", "1984", "To Kill a Mockingbird"]
embeddings = np.random.rand(3, 128).astype(np.float32)

collection.insert([ids, titles, embeddings])
```

**行式组织（字典列表）**：
```python
data = [
    {"id": 101, "title": "The Great Gatsby", "embedding": [0.1, 0.2, ...]},
    {"id": 102, "title": "1984", "embedding": [0.3, -0.1, ...]}
]
collection.insert(data)
```

**Pandas DataFrame**：
```python
import pandas as pd

df = pd.DataFrame({
    "id": [101, 102, 103],
    "title": ["Gatsby", "1984", "Mockingbird"],
    "embedding": [[0.1, 0.2, ...], [0.3, -0.1, ...], [0.5, 0.6, ...]]
})
collection.insert(df)
```

**持久化**：
```python
# 强制将内存中的数据持久化到磁盘
collection.flush()
```

**插入参数**：
- `partition_name`：指定插入到哪个分区
- `timeout`：超时时间

**返回值 MutationResult**包含：
- `insert_count`：插入数量
- `primary_keys`：插入实体的主键列表
- `timestamp`：操作时间戳

---

### 5. 索引创建

索引用于加速向量检索。

#### 5.1 使用 MilvusClient 创建索引

```python
# 准备索引参数
index_params = client.prepare_index_params()

# 添加索引
index_params.add_index(
    field_name="embedding",
    index_type="IVF_FLAT",      # 索引类型
    metric_type="L2",           # 度量类型
    params={"nlist": 128}       # 索引参数
)

# 创建索引
client.create_index(
    collection_name="products",
    index_params=index_params
)
```

#### 5.2 使用 ORM 创建索引

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)
```

#### 5.3 常用索引类型

| 索引类型 | 适用场景 | 说明 |
|---------|---------|------|
| **FLAT** | <10万条数据 | 暴力搜索，100%召回率，适合算法验证基线 |
| **IVF_FLAT** | 百万级数据 | 倒排索引，`nlist`建议设为√N |
| **IVF_SQ8** | 百万级数据 | 量化压缩，内存占用更小 |
| **HNSW** | 千万级数据 | 图索引，参数`M`（连接边数）和`efConstruction` |
| **AUTOINDEX** | 通用 | 由 Milvus 自动选择最优索引 |

**关键参数**：
- `nlist`：聚类中心数量（IVF 系列）
- `nprobe`：查询时访问的聚类中心数（影响精度与速度平衡）
- `M`：HNSW 每个节点的连接边数，默认16
- `efConstruction`：HNSW 建图时的候选邻域大小，默认200

---

### 6. 数据加载与搜索

**⚠️ 重要**：搜索前必须先加载集合到内存。

```python
collection.load()
```

#### 6.1 相似性搜索

**使用 MilvusClient**：
```python
# 准备查询向量
query_vector = [0.1, 0.2, 0.3, ...]  # 维度需与集合一致

# 执行搜索
results = client.search(
    collection_name="products",
    data=[query_vector],              # 支持批量查询
    anns_field="embedding",           # 向量字段名
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,                         # 返回 top-K 结果
    output_fields=["id", "title"]     # 返回的标量字段
)

# 处理结果
for result in results:
    for hit in result:
        print(f"ID: {hit['id']}, Score: {hit['distance']}")
```

**使用 ORM**：
```python
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 10}
}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    output_fields=["id", "title"]
)
```

#### 6.2 标量过滤查询（Query）

按标量字段条件查询实体：

```python
# 查询价格大于 20 的所有实体
results = client.query(
    collection_name="products",
    filter="price > 20",
    output_fields=["id", "title", "price"]
)

# 带分页查询
results = client.query(
    collection_name="products",
    filter="title like '%Gatsby%'",
    limit=10,
    offset=0
)
```

#### 6.3 带过滤的向量搜索（混合检索）

先按标量条件过滤，再在过滤结果中进行向量搜索：

```python
results = client.search(
    collection_name="products",
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=10,
    filter="price < 50",          # 标量过滤条件
    output_fields=["id", "title", "price"]
)
```

---

### 7. 分区管理（Partition）

分区是 Collection 的逻辑子集，用于组织和管理数据。

```python
# 创建分区
collection.create_partition("partition_2024")

# 列出所有分区
partitions = collection.partitions

# 插入数据到指定分区
collection.insert(data, partition_name="partition_2024")

# 搜索指定分区
collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10,
    partition_names=["partition_2024"]   # 指定分区
)

# 删除分区
collection.drop_partition("partition_2024")
```

---

### 8. 数据删除与更新

```python
# 按主键删除
client.delete(
    collection_name="products",
    filter="id in [1, 2, 3]"
)

# Upsert（插入或更新）
client.upsert(
    collection_name="products",
    data=[{"id": 1, "title": "New Title", "embedding": [...]}]
)
```

---


### 9. 列搜索


**第一层：核心匹配（距离计算）—— 只针对向量列**

向量搜索的本质是**算距离**（余弦相似度、内积、欧式距离）。

- **匹配对象**：你传入的“查询向量（Query Vector）”。
- **被匹配对象**：存储在**向量字段（如 `embeddings` 列）**里的所有向量。
- **计算过程**：系统会把你传入的向量，和表中每一行（Entity）的向量字段里的数值，逐一进行数学运算（算点积或距离），然后按相似度从高到低排序，返回Top K。

> **面试话术**：“在Milvus里，真正的‘相似度计算’发生在**向量字段（Vector Field）**上。比如我建了一个`book_embeddings`列存向量，搜索时，系统只拿我的查询向量去跟这一列的768个数值算距离，别的列不参与数学计算。”



**第二层：多向量场景（进阶）—— 搜索时必须指定“匹配哪个列”**

这是很多新手会踩的坑。Milvus 2.x 及以上版本支持**一个集合（表）里拥有多个向量字段**。

比如你有张“商品表”：
- 列A：`title_embedding`（标题向量，768维）
- 列B：`image_embedding`（图片向量，512维）

**问题来了**：如果你搜的时候不告诉它匹配哪个，它根本不知道用哪列算距离。

**解决方案**：在调用搜索API（`search`）时，必须显式传入 `vector_field_name` 参数。

```python
# Milvus Python SDK 伪代码
milvus_client.search(
    collection_name="products",
    data=[query_vector],           # 你传入的向量
    anns_field="title_embedding",  # 🔥 关键！明确指定匹配这个向量字段
    param={"metric_type": "COSINE"},
    limit=10
)
```

**第三层：字段过滤**

**重点来了**：其他列（如 `price`、`author`、`category`）**不参与**向量距离计算，但它们极其重要，通常用于**预过滤（Pre-filtering）**或**后过滤（Post-filtering）**。

比如你搜“推荐3本200元以下的AI书籍”：
1. **标量过滤**：先通过 `WHERE price < 200` 把不符合价格的行筛掉（利用倒排索引）。
2. **向量匹配**：只在上一步筛出来的小范围里，拿查询向量去匹配 `embeddings` 列，算距离排序。



---


## 二、LangChain + Milvus 集成

### 1. 安装

```bash
pip install -qU langchain-milvus milvus-lite langchain-openai
```

`langchain-milvus` 提供了 LangChain 与 Milvus 的无缝集成。

### 2. 初始化向量存储

```python
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus

# 使用 Milvus Lite（本地文件）
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
)
```

**使用 Milvus 服务器**：
```python
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "http://localhost:19530"},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)
```

### 3. 从文档创建集合

```python
from langchain_core.documents import Document

# 从文档创建
vector_store = Milvus.from_documents(
    documents=[Document(page_content="这是一段文本内容")],
    embedding=embeddings,
    collection_name="langchain_example",
    connection_args={"uri": URI},
)

# 加载已有集合
vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name="langchain_example",
)
```

### 4. 添加文档

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="今天早餐吃了巧克力薄煎饼和炒鸡蛋。",
        metadata={"source": "tweet"}
    ),
    Document(
        page_content="明天天气预报多云，最高气温62度。",
        metadata={"source": "news"}
    )
]

# 添加文档（自动生成向量并插入）
ids = vector_store.add_documents(documents=documents)

# 或指定 ID
ids = vector_store.add_documents(documents=documents, ids=["1", "2"])
```

### 5. 相似性搜索

```python
# 基本相似性搜索
results = vector_store.similarity_search(
    query="早餐吃了什么？",
    k=2
)
for doc in results:
    print(doc.page_content)

# 带相似度分数的搜索
results = vector_store.similarity_search_with_score(
    query="天气预报",
    k=2
)
for doc, score in results:
    print(f"Score: {score}, Content: {doc.page_content}")

# 带过滤条件的搜索
results = vector_store.similarity_search(
    query="新闻",
    k=2,
    filter={"source": "news"}  # 只搜索 source 为 news 的文档
)
```

### 6. 删除文档

```python
# 按 ID 删除
vector_store.delete(ids=["1", "2"])
```

### 7. 异步操作

```python
# 异步添加
await vector_store.aadd_documents(documents=documents, ids=ids)

# 异步搜索
results = await vector_store.asimilarity_search(query="测试", k=2)

# 异步带分数搜索
results = await vector_store.asimilarity_search_with_score(query="测试", k=2)
```

### 8. 作为检索器使用

```python
# 转换为检索器
retriever = vector_store.as_retriever(
    search_type="similarity",  # 或 "mmr" (最大边际相关性)
    search_kwargs={"k": 3}
)

# 使用检索器
docs = retriever.get_relevant_documents("查询文本")
for doc in docs:
    print(doc.page_content)
```

**MMR 检索**（提高结果多样性）：
```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

### 9. 混合搜索（Hybrid Search）

Milvus 支持密集向量 + 稀疏向量（如 BM25）的混合搜索。

```python
from langchain_milvus import MilvusCollectionHybridSearchRetriever

# 创建混合搜索检索器
retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    # 配置多个向量字段的搜索参数
)
```

### 10. 完整 RAG 示例

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 创建检索器
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 创建 Prompt 模板
prompt = PromptTemplate(
    template="""基于以下上下文回答问题：
    上下文：{context}
    问题：{question}
    回答：""",
    input_variables=["context", "question"]
)

# 创建 RAG 链
llm = ChatOpenAI(model="gpt-4")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 执行问答
result = qa_chain.invoke("你的问题")
print(result["result"])
```

### 11. Milvus 配置参数详解

初始化 `Milvus` 向量存储时的关键参数：

| 参数 | 说明 |
|------|------|
| `collection_name` | 集合名称，默认 "LangChainCollection" |
| `collection_description` | 集合描述 |
| `connection_args` | 连接参数，如 `{"uri": "..."}` |
| `consistency_level` | 一致性级别，默认 "Session" |
| `index_params` | 索引参数配置 |
| `search_params` | 搜索参数配置 |
| `drop_old` | 是否删除已有同名集合 |
| `auto_id` | 是否自动生成主键 ID |
| `primary_field` | 主键字段名 |
| `text_field` | 文本字段名 |
| `vector_field` | 向量字段名 |
| `enable_dynamic_field` | 是否启用动态字段 |

---

## 三、操作流程总结

Milvus 核心操作的正确顺序：

```
1. 连接 Milvus 服务
2. 创建 Collection（定义 Schema）
3. 插入数据
4. 创建索引
5. 加载 Collection 到内存
6. 执行搜索/查询
```
