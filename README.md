# 上财校园助手 · CampusAgent

面向上海财经大学学生的校园问答智能体，基于学校及学院官方规章制度构建检索增强生成（RAG）应用。学生输入问题，系统检索官方资料并生成带来源标注的回答。

## 功能

- **官方资料问答**：知识库来自 `规章制度/` 下的官方文件（学工、本科/研究生教学、科研、财务、党建等约 70 份），回答严格依据资料、禁止编造。
- **来源可追溯**：每个回答附来源卡片（文件名 + 分类 + 原文片段），可点击查看原文全文。
- **检索方式**：向量语义检索（可选）+ BM25 关键词检索，无向量依赖时自动降级。
- **流式输出**：回答逐字流式生成，交互友好。
- **多轮对话**：支持连续追问，模型结合上文理解指代（如"它需要什么材料"）。
- **分类限定检索**：侧栏点击知识库分类后，仅在该分类下检索，答案更聚焦、更准确。
- **敏感信息过滤**：入库前自动扫描并剔除含手机号、身份证号、银行卡号、明文账号密码的内容。

## 技术栈

Python 3.11+ · FastAPI · ChromaDB + sentence-transformers（可选）· rank-bm25 + jieba · OpenAI 兼容接口（dsv4f）

## 本地运行

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
# （可选）语义检索：
# pip install -r requirements-vector.txt
cp .env.example .env          # 填入 OPENAI_BASE_URL / OPENAI_API_KEY

# 构建知识库索引（解析 → 分块 → 索引）
python -m ingest.parse
python -m ingest.chunk
python -m ingest.embed_store

# 启动服务
uvicorn app.main:app --port 8000
```

浏览器打开 <http://localhost:8000>。

## 部署（Hugging Face Space）

1. 在 <https://huggingface.co/new-space> 创建 Space，SDK 选择 **Docker**。
2. 推送本仓库到该 Space（或关联 Git）。
3. 在 Space Settings → Variables and secrets 添加：
   - `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL=dsv4f`
4. 构建完成后打开 Space 的 Public App 链接即可体验。

索引在 Docker 构建期生成并打包进镜像（BM25）；文档更新后需重建 Space。

## 项目结构

```
backend/
  ingest/        # 离线：解析、敏感过滤、分块、建索引
  app/           # FastAPI：/api/chat（SSE）、/api/meta、/api/doc
  data/          # 生成产物（gitignore）
frontend/        # 单页 Web 界面
规章制度/        # 知识源（上海财经大学官方文件）
```
