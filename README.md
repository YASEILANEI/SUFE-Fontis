# 金匮问渠 · SUFE Fontis

面向上海财经大学学生的校园问答智能体，基于学校及学院官方规章制度构建检索增强生成（RAG）应用。学生输入问题，系统检索官方资料并生成带来源标注的回答。

## 功能

- **官方资料问答**：知识库来自 `规章制度/` 下的 80 份官方文件（学工、本科教学、研究生教学、科研、学校财务规定、党建、出国出境等），回答严格依据资料、禁止编造。
- **来源可追溯**：每个回答附来源卡片（文件名 + 分类 + 原文片段），点击可在原文中定位并高亮引用区间，也可下载原文件。
- **混合检索**：向量语义检索（可选）+ BM25 关键词检索，按加权分数合并去重；未安装向量依赖时自动降级为 BM25。
- **流式输出**：回答逐字流式生成（SSE），交互友好。
- **多轮对话**：支持连续追问，模型结合上文理解指代（如"它需要什么材料"）。
- **分类限定检索**：侧栏点击知识库分类后，仅在该分类下检索，答案更聚焦。
- **敏感信息过滤**：入库前自动剔除含手机号、身份证号、银行卡号、明文账号密码的内容。

## 技术栈

Python 3.11+ · FastAPI · rank-bm25 + jieba · ChromaDB + sentence-transformers（可选）· OpenAI 兼容接口

## 快速开始

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows（macOS/Linux: source .venv/bin/activate）
pip install -r requirements.txt
# （可选）启用向量语义检索：
# pip install -r requirements-vector.txt
cp .env.example .env          # 填入 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL

# 构建知识库索引（解析 → 分块 → 建索引）
python -m ingest.parse
python -m ingest.chunk
python -m ingest.embed_store

# 启动服务
uvicorn app.main:app --port 8000
```

浏览器打开 <http://localhost:8000>。

> 说明：`.doc` 等旧格式需先转换才能解析；扫描件（无文字层 PDF，`note="scan"`）需 OCR 后才能入库，索引时自动跳过。

## API

- `GET /api/meta` —— 知识库统计、分类列表、首页引导问题
- `POST /api/chat` —— 问答（SSE 流式，事件序列 `sources` → `delta*` → `done`）
- `GET /api/doc/{id}` —— 原文全文，`?cite=` 指定引用块号时返回高亮区间
- `GET /api/doc/{id}/download` —— 下载原始文件

## 部署（Hugging Face Space）

1. 在 <https://huggingface.co/new-space> 创建 Space，SDK 选择 **Docker**。
2. 推送本仓库到该 Space（或关联 Git）。
3. 在 Space Settings → Variables and secrets 添加 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、`OPENAI_MODEL`。
4. 构建完成后打开 Space 的 Public App 链接即可体验。

索引在 Docker 构建期生成并打包进镜像（BM25）；更新 `规章制度/` 下的文档后需重建 Space。

## 项目结构

```
backend/
  ingest/        # 离线流水线：解析、敏感过滤、分块、建索引
  app/           # FastAPI：/api/chat（SSE）、/api/meta、/api/doc
  data/          # 生成产物（gitignore）
  tests/         # pytest 测试
frontend/        # 单页 Web 界面（index.html）
规章制度/        # 知识源（上海财经大学官方文件，按分类子目录组织）
```
