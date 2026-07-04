
# Milvus 基础概念与实战指南

## 一、Milvus 基础概念

**1. 什么是向量数据库？**

- **传统数据库**：存储结构化数据（如表格），通过 SQL 查询。
- **向量数据库**：存储高维向量（如文本、图像、音频的嵌入向量），通过相似度算法（如余弦相似度、欧氏距离）快速检索。
- **应用场景**：推荐系统、智能搜索、异常检测、知识图谱等。

---

**2. Milvus 核心组件**

| 组件 | 功能 |
|------|------|
| **Coord Node（协调节点）** | 管理集群元数据和任务调度。 |
| **Query Node（查询节点）** | 处理查询请求，执行向量检索。 |
| **Data Node（数据节点）** | 存储向量数据和索引，处理数据插入/删除。 |
| **Index Node（索引节点）** | 构建和管理向量索引（2.x 版本后合并到 Data Node）。 |
| **Proxy（代理）** | 提供 API 接口，客户端通过 Proxy 与 Milvus 交互。 |

**Storage（存储层）**：
- **Meta Storage（元数据存储）**：使用 ETCD 或 MySQL 存储集群状态。
- **Object Storage（对象存储）**：使用 MinIO 或 S3 存储向量数据和索引。
- **Log Storage（日志存储）**：使用本地磁盘或云存储保存日志。

---

**3. Milvus 版本选择**

- **Milvus 2.x**：分布式架构，支持高可用和弹性扩展（**推荐生产环境使用**）。
- **Zilliz Cloud**：全托管 Milvus 服务，免运维（适合快速上手）。

---

## 二、单机部署（Docker 快速上手）

**1. 环境准备**

**硬件要求**：
- CPU：4 核以上（推荐 8 核）
- 内存：16 GB 以上（推荐 32 GB）
- 磁盘：SSD（至少 100 GB 可用空间）

**软件依赖**：
- Docker：最新版本（[安装教程](https://docs.docker.com/get-docker/)）
- Docker Compose：最新版本（[安装教程](https://docs.docker.com/compose/install/)）

---

**2. 通过 Docker Compose 部署**
1. **下载配置文件**：
   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
   ```

2. **启动服务**：
   ```bash
   docker-compose up -d
   ```

3. **验证服务**：
   ```bash
   docker-compose ps
   ```

4. **检查容器状态**：
   ```bash
   docker ps
   ```

5. **访问 Web 界面（可选）**：
   打开浏览器访问 `http://localhost:9091`（默认端口）。

---

**3. 通过 Docker 直接部署**

1. **拉取镜像**：
   ```bash
   docker pull milvusdb/milvus:latest
   ```

2. **启动容器**：
   ```bash
   docker run -d --name milvus_standalone \
     -p 19530:19530 \
     -p 9091:9091 \
     -v /path/to/data:/var/lib/milvus \
     milvusdb/milvus:latest
   ```

   - `-p 19530:19530`：gRPC 接口（客户端连接端口）
   - `-p 9091:9091`：HTTP 接口（Web 界面）
   - `-v /path/to/data:/var/lib/milvus`：数据持久化目录

---

## 三、Python 客户端快速入门

### 1. 安装 PyMilvus

```bash
pip install pymilvus
```

### 2. 连接 Milvus

```python
from pymilvus import connections

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)
```

### 3. 创建集合（Collection）

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

fields = [
    # 1. fields (必填): 定义了集合的4个字段
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # 主键，自动生成
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128)
    
    description="商品信息集合",
    # 3. auto_id (可选): 此处未设置，但在 FieldSchema 中已对 id 字段开启
    # 4. 其他参数如 enable_dynamic_field, primary_field 等可根据需要添加
]
schema = CollectionSchema(
                      fields,      # 1. fields (必填): 定义了集合的4个字段
                      description="example collection"      # 2. description (可选): 为集合添加描述
)

collection = Collection("my_collection", schema)
```

### 4. 插入数据

```python
import random

entities = [
    [i for i in range(1000)],                     # id
    [[random.random() for _ in range(128)] for _ in range(1000)]  # embedding
]
collection.insert(entities)
```

### 5. 创建索引

```python
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 8, "efConstruction": 200}
}
collection.create_index("embedding", index_params)
collection.load()
```

### 6. 向量检索

```python
# 生成查询向量
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}
}

# 执行相识度搜索
query_vector = [random.random() for _ in range(128)]
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10
)

# 打印
for hits in results:
    for hit in hits:
        print(f"id: {hit.id}, score: {hit.score}")
```

---

## 四、常见问题与解决方案

### 1. 部署问题

**Q1：Docker 容器启动失败**
- **原因**：端口冲突或配置错误。
- **解决**：检查端口占用（`netstat -tulnp | grep 19530`），修改 `docker-compose.yml` 中的端口映射。

**Q2：连接失败（Connection refused）**
- **原因**：Milvus 服务未启动或端口错误。
- **解决**：检查容器状态（`docker ps`），确认端口映射正确。

---

### 2. 性能问题

**Q3：检索速度慢**
- **原因**：未创建索引或索引类型不合适。
- **解决**：为向量字段创建索引（如 HNSW 或 IVF_FLAT），调整参数（如 `efConstruction`）。

**Q4：内存不足**
- **原因**：数据量过大或索引未压缩。
- **解决**：增加容器内存限制，或使用量化索引（如 SQ8）。

---

### 3. 数据问题

**Q5：插入数据失败**
- **原因**：向量维度不匹配或主键重复。
- **解决**：检查向量维度（`dim` 参数），确保主键唯一。

**Q6：数据丢失**
- **原因**：未持久化或意外删除容器。
- **解决**：使用挂载卷（`-v /host/path:/var/lib/milvus`）持久化数据，定期备份。
```
