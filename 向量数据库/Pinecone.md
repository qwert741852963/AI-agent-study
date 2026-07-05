# Pinecone 向量数据库完整使用指南（Python SDK + LangChain 集成）


## 一、环境准备与初始化

### 1.1 安装

```bash
pip install pinecone
```

如需开发依赖（测试、类型检查等）：
```bash
pip install pinecone[dev]
```

### 1.2 初始化客户端

```python
from pinecone import Pinecone, ServerlessSpec

# 方式一：直接传入 API Key
pc = Pinecone(api_key="your-api-key")

# 方式二：从环境变量 PINECONE_API_KEY 读取
pc = Pinecone()

# 自定义控制平面主机
pc = Pinecone(api_key="your-api-key", host="https://api.pinecone.io")

# 配置超时时间（秒）
pc = Pinecone(api_key="your-api-key", timeout=30)
```



### 1.3 异步客户端

```python
import asyncio
from pinecone import AsyncPinecone

async def main():
    async with AsyncPinecone(api_key="your-api-key") as pc:
        # 异步操作...
        pass

asyncio.run(main())
```




## 二、索引（Index）管理

### 2.1 创建索引

**创建 Serverless 索引（推荐）：**

```python
pc.create_index(
    name="my-index",
    dimension=1536,           # 向量维度，需与嵌入模型匹配
    metric="cosine",          # 相似度度量：cosine / euclidean / dotproduct
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
```

**创建 Pod-based 索引：**

```python
from pinecone import PodSpec

pc.create_index(
    name="my-pod-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east1-gcp",
        pod_type="p1.x1",
        replicas=1
    )
)
```

**支持的相似度度量：**
- `cosine`：余弦相似度
- `euclidean`：欧几里得距离
- `dotproduct`：点积

**索引类型说明：**
- **密集向量索引（Dense）** ：用于语义搜索，存储稠密向量嵌入
- **稀疏向量索引（Sparse）** ：用于词法搜索，使用稀疏嵌入模型
- **文档模式索引（Document Schema）** ：支持密集向量、稀疏向量和全文搜索字段混合

### 2.2 列出索引

```python
# 列出所有索引
indexes = pc.list_indexes()
for idx in indexes:
    print(idx.name)
```



### 2.3 检查索引是否存在

```python
exists = pc.has_index("my-index")
print(exists)  # True / False
```

### 2.4 描述索引

```python
desc = pc.describe_index("my-index")
print(desc.name)
print(desc.dimension)
print(desc.metric)
print(desc.status)  # Ready / Initializing
```

### 2.5 配置索引

```python
# 启用删除保护、添加标签等
pc.configure_index(
    name="my-index",
    deletion_protection=True,
    tags={"env": "production"}
)
```



### 2.6 删除索引

```python
pc.delete_index("my-index")
```



### 2.7 连接到索引

```python
# 通过名称连接
index = pc.Index("my-index")

# 通过主机地址连接（性能更优）
index = pc.Index(host="my-index-abc123.svc.pinecone.io")
```


## 三、向量操作（核心 CRUD）

### 3.1 插入/更新向量（Upsert）

**基本用法：**

```python
index.upsert(
    vectors=[
        ("id-1", [0.1, 0.2, 0.3, ...]),           # (ID, 向量值)
        ("id-2", [0.4, 0.5, 0.6, ...]),
    ],
    namespace="my-namespace",      # 可选，默认为空字符串
    batch_size=100                 # 自动分批，避免内存问题
)
```

**带元数据的 Upsert：**

```python
index.upsert(
    vectors=[
        ("id-1", [0.1, 0.2, ...], {"category": "science", "source": "wiki"}),
        ("id-2", [0.4, 0.5, ...], {"category": "art", "source": "blog"}),
    ],
    namespace="my-namespace"
)
```

**使用 Vector 对象：**

```python
from pinecone import Vector

index.upsert(
    vectors=[
        Vector(id="id-1", values=[0.1, 0.2, ...], metadata={"key": "value"}),
    ]
)
```



> **注意**：如果 ID 已存在，Upsert 会覆盖原有向量和元数据。

### 3.2 查询（Query）

**按向量查询：**

```python
results = index.query(
    vector=[0.1, 0.2, 0.3, ...],   # 查询向量
    top_k=10,                       # 返回最相似的 K 个结果
    namespace="my-namespace",       # 可选
    include_values=True,            # 是否返回向量值
    include_metadata=True           # 是否返回元数据
)

for match in results.matches:
    print(f"ID: {match.id}, Score: {match.score}")
    print(f"Metadata: {match.metadata}")
```

**按 ID 查询（使用已存储的向量作为查询）：**

```python
results = index.query(
    id="existing-vector-id",
    top_k=10,
    namespace="my-namespace"
)
```

> 每个查询请求只能包含 `id` 或 `vector` 中的一个参数。

### 3.3 获取向量（Fetch）

```python
# 按 ID 获取向量
vectors = index.fetch(
    ids=["id-1", "id-2", "id-3"],
    namespace="my-namespace"
)

for vector_id, vector in vectors.vectors.items():
    print(f"ID: {vector_id}")
    print(f"Values: {vector.values}")
    print(f"Metadata: {vector.metadata}")
```



### 3.4 删除向量

**按 ID 删除：**

```python
index.delete(
    ids=["id-1", "id-2"],
    namespace="my-namespace"
)
```

**删除命名空间中的所有向量（保留命名空间）：**

```python
index.delete(
    delete_all=True,
    namespace="my-namespace"
)
```

**按元数据过滤删除：**

```python
index.delete(
    filter={"category": {"$eq": "old_data"}},
    namespace="my-namespace"
)
```

### 3.5 更新向量

> **注意**：Pinecone 没有单独的 `update` 操作。更新向量使用 `upsert`，传入相同的 ID 即可覆盖。

```python
# 更新向量值
index.upsert(
    vectors=[("id-1", [0.9, 0.8, 0.7, ...], {"category": "updated"})],
    namespace="my-namespace"
)
```


## 四、命名空间（Namespace）管理

命名空间用于在同一个索引中隔离数据，实现多租户隔离。查询时一个命名空间不会返回另一个命名空间的结果。

### 4.1 默认命名空间

如果不指定 `namespace` 参数，Pinecone 使用默认的空字符串 `""` 作为命名空间。

### 4.2 删除命名空间（仅 Serverless 索引）

```python
index.delete_namespace("my-namespace")
```

> **警告**：此操作不可逆，命名空间中的所有数据将被永久删除。

### 4.3 命名空间注意事项

- Pinecone **不支持直接重命名**命名空间
- 如需重命名，需将数据从旧命名空间读取并写入新命名空间


## 五、集合（Collection）管理

集合是 Pod-based 索引的只读快照，用于备份、复制或恢复索引。**Serverless 索引使用备份机制而非集合**。

### 5.1 创建集合

```python
collection = pc.create_collection(
    name="snapshot-2025-01",
    source="my-pod-index"        # 源索引名称
)
print(collection.status)  # "Initializing"
```



### 5.2 列出集合

```python
for col in pc.list_collections():
    print(col.name, col.status)

# 仅获取名称列表
names = pc.list_collections().names()
```



### 5.3 描述集合

```python
col = pc.describe_collection("snapshot-2025-01")
print(col.name)          # "snapshot-2025-01"
print(col.status)        # "Ready"
print(col.dimension)     # 向量维度
print(col.vector_count)  # 向量数量
print(col.size)          # 大小（字节）
```



### 5.4 从集合创建索引

```python
from pinecone import PodSpec

pc.create_index(
    name="restored-index",
    dimension=1536,
    metric="cosine",
    spec=PodSpec(
        environment="us-east-1-aws",
        pod_type="p1.x1",
        source_collection="snapshot-2025-01"
    )
)
```

新索引将预填充集合快照中的所有向量。

### 5.5 删除集合

```python
pc.delete_collection("snapshot-2025-01")
```

## 六、索引、命名空间、集合关系
## 6.1 关系间区别

| 概念 | 层级 / 作用 | 关键特性 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **索引 (Index)** | 最高数据组织单位 | 独立计算资源，定义维度/度量 | 区分不同应用或完全不同类型的数据 |
| **命名空间 (Namespace)** | 索引内的逻辑分区 | **完全隔离**，轻量级 | 多租户隔离、按环境/语言分区 |
| **集合 (Collection)** | 索引的静态备份（仅Pod索引） | **只读、不可查询**，用于备份/恢复 | 备份索引、复制数据、恢复索引 |

## 与Milvus的区别


简单来说，可以用一个公式来理解：
**Milvus 的 `Collection` ≈ Pinecone 的 `Index`**
**Milvus 的 `Partition` ≈ Pinecone 的 `Namespace`**

### 🔎 核心概念对比

| 概念层级 | Pinecone | Milvus | 说明 |
| :--- | :--- | :--- | :--- |
| **最高层级数据容器** | **索引 (Index)** | **集合 (Collection)** | 两者功能对等，都是最顶层的逻辑单元，类似于关系型数据库中的“表”。 |
| **容器内的逻辑划分** | **命名空间 (Namespace)** | **分区 (Partition)** | 两者都用于在数据容器内进行逻辑隔离。 |
| **最小数据单元** | **记录 (Record)** | **实体 (Entity)** | 两者都代表一条具体的数据记录。 |
| **特殊概念** | **集合 (Collection)** | **备份 (Backup)** | Pinecone中的“集合”是索引的静态快照，仅用于备份和恢复**（不可查询）** 。 |


## 六、元数据过滤查询

### 6.1 基本过滤

```python
results = index.query(
    vector=[0.1, 0.2, ...],
    top_k=10,
    filter={"category": {"$eq": "science"}},
    namespace="my-namespace"
)
```



### 6.2 支持的过滤操作符

| 操作符 | 说明 | 示例 |
|--------|------|------|
| `$eq` | 等于 | `{"genre": {"$eq": "fiction"}}` |
| `$ne` | 不等于 | `{"genre": {"$ne": "fiction"}}` |
| `$gt` | 大于 | `{"score": {"$gt": 0.8}}` |
| `$gte` | 大于等于 | `{"score": {"$gte": 0.8}}` |
| `$lt` | 小于 | `{"score": {"$lt": 0.5}}` |
| `$lte` | 小于等于 | `{"score": {"$lte": 0.5}}` |
| `$in` | 在列表中 | `{"genre": {"$in": ["fiction", "poetry"]}}` |
| `$nin` | 不在列表中 | `{"genre": {"$nin": ["fiction"]}}` |



### 6.3 组合过滤

```python
# AND 逻辑（隐式）
filter = {
    "category": {"$eq": "science"},
    "year": {"$gt": 2020}
}

# OR 逻辑
filter = {
    "$or": [
        {"category": {"$eq": "science"}},
        {"category": {"$eq": "art"}}
    ]
}
```


## 七、LangChain 集成

### 7.1 安装

```bash
pip install -qU langchain langchain-pinecone langchain-openai
```



配置环境变量：
```bash
export PINECONE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-openai-key"  # 可选，使用 OpenAI 嵌入时需要
```



### 7.2 初始化 PineconeVectorStore

**方式一：传入已有索引：**

```python
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-index")

vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
```

**方式二：通过索引名称（自动创建）：**

```python
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = PineconeVectorStore(
    index_name="my-index",
    embedding=OpenAIEmbeddings(),
    pinecone_api_key="your-api-key"  # 或通过环境变量
)
```



**方式三：直接使用主机地址（异步优化）：**

```python
vector_store = PineconeVectorStore(
    host="my-index-abc123.svc.pinecone.io",
    embedding=OpenAIEmbeddings(),
    pinecone_api_key="your-api-key"
)
```

### 7.3 初始化参数说明

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `index` | Optional[Any] | 否* | None | 预创建的 Pinecone 索引 |
| `embedding` | Optional[Embeddings] | 是 | None | 嵌入函数 |
| `text_key` | Optional[str] | 否 | "text" | 存储文档文本的元数据键 |
| `namespace` | Optional[str] | 否 | None | 默认命名空间 |
| `distance_strategy` | Optional[DistanceStrategy] | 否 | COSINE | 相似度度量 |
| `pinecone_api_key` | Optional[str] | 否* | 从环境变量 | Pinecone API Key |
| `index_name` | Optional[str] | 否* | 从环境变量 | 索引名称 |
| `host` | Optional[str] | 否* | 从环境变量 | 直接索引主机地址 |

> *至少需要提供 `index`、`index_name` 或 `host` 中的一个。

### 7.4 添加文档（add_documents）

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="这是文档内容",
        metadata={"source": "article", "category": "science"}
    ),
    # 更多文档...
]

# 添加文档并自动生成嵌入
vector_store.add_documents(documents)

# 或添加文本（自动生成 ID）
ids = vector_store.add_texts(
    texts=["文本1", "文本2"],
    metadatas=[{"source": "a"}, {"source": "b"}]
)
```



### 7.5 相似性搜索

**基本相似性搜索：**

```python
docs = vector_store.similarity_search(
    query="什么是机器学习？",
    k=5  # 返回结果数量
)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

**带相似度分数的搜索：**

```python
docs_with_scores = vector_store.similarity_search_with_score(
    query="什么是机器学习？",
    k=5
)

for doc, score in docs_with_scores:
    print(f"Score: {score}")
    print(doc.page_content)
```

**带元数据过滤的搜索：**

```python
docs = vector_store.similarity_search(
    query="什么是机器学习？",
    k=5,
    filter={"category": {"$eq": "science"}}
)
```

**最大边际相关性搜索（MMR）：**

```python
docs = vector_store.max_marginal_relevance_search(
    query="什么是机器学习？",
    k=5,
    fetch_k=20  # 先获取20个，再从中选择多样化的5个
)
```

### 7.6 异步操作

```python
# 异步相似性搜索
docs = await vector_store.asimilarity_search(
    query="什么是机器学习？",
    k=5
)

# 异步添加文档
await vector_store.aadd_documents(documents)
```

### 7.7 稀疏向量支持（PineconeSparseVectorStore）

对于使用稀疏嵌入模型（如 SPLADE）的场景：

```python
from langchain_pinecone import PineconeSparseVectorStore
from langchain_pinecone.sparse_embeddings import PineconeSparseEmbeddings

sparse_store = PineconeSparseVectorStore(
    index_name="my-sparse-index",
    embedding=PineconeSparseEmbeddings(model="pinecone-sparse-english-v0")
)

# 使用方式与密集向量类似
docs = sparse_store.similarity_search("查询文本", k=5)
```



### 7.8 性能优化建议

对于 OpenAI 嵌入，建议配置：
- `pool_threads > 4`
- `embedding_chunk_size > 1000`
- `batch_size ≈ 64`

```python
index = pc.Index("my-index", pool_threads=8)

vector_store = PineconeVectorStore(
    index=index,
    embedding=OpenAIEmbeddings(),
    # add_documents 时 batch_size 默认为 64
)
```


## 八、CLI 快速参考

Pinecone CLI 也可用于管理操作：

```bash
# 列出索引
pc index list

# 创建索引
pc index create -n my-index -d 1536 -m cosine -c aws -r us-east-1

# 描述索引
pc index describe --name my-index

# 删除索引
pc index delete --name my-index

# 删除命名空间
pc index namespace delete -n my-index --name tenant-a
```


## 九、重要注意事项

1. **Serverless vs Pod-based**：
   - Serverless：按需付费，自动扩展，推荐用于开发和中小规模应用
   - Pod-based：预留资源，性能稳定，适合大规模生产环境

2. **集合（Collection）仅支持 Pod-based 索引**

3. **命名空间删除仅支持 Serverless 索引**

4. **更新操作使用 Upsert**：Pinecone 没有独立的 update 操作，传入相同 ID 即可覆盖

5. **批量操作建议**：使用 `batch_size` 参数自动分批，避免内存问题

6. **调试模式**：设置环境变量 `PINECONE_DEBUG=1` 启用调试日志
