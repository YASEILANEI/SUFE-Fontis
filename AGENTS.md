# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 项目概述

金匮问渠（SUFE Fontis）：面向上海财经大学学生的 RAG 问答应用，知识库来自 `规章制度/` 下的 80 份官方文件（党建、学工、本科教学、研究生教学、科研、学校财务规定、出国出境规章制度，按一级子目录分类）。回答严格依据检索到的资料、禁止编造，并附来源标注，前端可定位原文并高亮引用区间。

技术栈：Python 3.11 · FastAPI · rank-bm25 + jieba · ChromaDB + sentence-transformers（可选，未安装自动降级 BM25）· OpenAI 兼容接口（模型经 `OPENAI_MODEL` 配置）。

## 常用命令（均在 `backend/` 目录下执行）

```bash
# 启动服务（前端 index.html 由 FastAPI 静态托管，浏览器开 http://localhost:8000）
uvicorn app.main:app --port 8000

# 建索引：新增/修改 规章制度/ 下文档后必须完整重跑这三步（顺序依赖）
python -m ingest.parse          # 规章制度/ → data/parsed/*.txt + data/documents.json
python -m ingest.chunk          # → data/chunks.json（分块 + 敏感信息过滤）
python -m ingest.embed_store    # → data/bm25_tokens.json + data/index_ready.json（可选 chroma/）

# 测试（pytest 未装在 requirements 里，需先 pip install pytest）
python -m pytest tests/
```

环境变量在 `backend/.env`（gitignored，模板见 `.env.example`）：`OPENAI_BASE_URL`（默认 `https://api.deepseek.com`）/ `OPENAI_API_KEY` / `OPENAI_MODEL`（默认 `deepseek-chat`）/ `EMBED_MODEL`。未配置 API key 时服务可启动但 `/api/chat` 会报错。

## 数据流与产物

`backend/data/` 是 ingest 生成的构建产物（gitignored），运行时 `Retriever` 以模块级单例加载并缓存：

- `parsed/<safe_id>.txt` —— 每个文档的纯文本
- `documents.json` —— 文档元数据 `[{id, title, category, rel_path, format, pages, note}]`
- `chunks.json` —— 检索单元 `[{id, doc_id, title, category, rel_path, heading, text, idx}]`
- `bm25_tokens.json` —— 预分词的 BM25 索引，运行时重建 BM25Okapi
- `index_ready.json` —— 状态标记 `{vector, bm25, chunks}`
- `chroma/` —— ChromaDB 持久化（仅装 `requirements-vector.txt` 后存在）

## 关键约定

- **ID 含斜杠**：`doc_id` 形如 `学工/xxx`；文件路径必须用安全形式 `safe_id = doc_id.replace("/", "__")`；`chunk_id` 形如 `{doc_id}#{idx}`。
- **ingest 顺序敏感**：`chunk.py` 依赖 `parse.py` 的输出，`embed_store.py` 依赖 `chunks.json`；扫描件（无文字层的 PDF，`note="scan"`）和 `.doc` 老格式在解析/分块阶段被跳过（需转换或 OCR 才能入库）。
- **检索合并**（`app/search.py`）：向量与 BM25 各取 top k 后按名次加权（向量 `(k-rank)*2`、BM25 `(k-rank)`）合并去重取 top k（默认 6）；分类限定在向量侧用 `where` 过滤、BM25 侧为取分后过滤。
- **SSE 协议**（`/api/chat`，POST）：事件序列为 `sources` → `delta`（可多个）→ `done`；任一步失败发 `error` 事件，前端不依赖 HTTP 状态码。
- **多轮对话**：`qa.py` 的 `build_retrieval_query` 把最近 ~6 条 user 消息拼进检索词；`stream_answer` 取最近 8 条消息作为 LLM 上下文。`/api/doc/{doc_id}` 返回原文全文，带 `?cite=` 参数时用 `citations.py` 在原文中定位各 chunk 的字节区间（支持空白折叠与重叠合并），供前端做引用高亮。

## 架构分层

- `backend/ingest/` —— 离线流水线：`parse.py`（PDF/DOCX/XLSX/XLS → 文本）→ `scan_sensitive.py`（手机号/身份证/银行卡/明文口令，严格命中整块丢弃）→ `chunk.py`（按 `第X章/第X条/一、/1./（一）` 等锚点切块，单块目标 ≤700 字）→ `embed_store.py`。
- `backend/app/` —— 运行时：`main.py`（路由与 SSE 装配，含首页引导问题 `SUGGESTED_QUESTIONS`）、`qa.py`（系统提示词 + 拼 prompt + 流式生成）、`search.py`（Retriever）、`citations.py`（来源定位）。
- `frontend/index.html` —— 单文件前端（marked + motion 走 CDN），通过 `/api/meta`、`/api/chat`（SSE）、`/api/doc/{id}`（`?cite=` 高亮）、`/api/doc/{id}/download` 交互，侧栏分类来自 `/api/meta`。
- `backend/tests/` —— pytest 测试（当前 `test_citations.py`，验证引用定位逻辑）。
- 部署为 Render Web Service（原生 buildpack，无需 Docker）：根目录 `render.yaml`（Blueprint）定义构建与启动命令；构建期在 Render 上跑完整 ingest（parse → chunk → embed_store），启动命令 `uvicorn app.main:app --port $PORT`，`OPENAI_BASE_URL` / `OPENAI_MODEL` 默认使用海外可访问的 DeepSeek 官方接口，`OPENAI_API_KEY` 走 Render Service Environment（`sync: false`）。ingest 与运行时均以 `Path(__file__).resolve().parents[2]` 定位仓库根，故 cwd 无关，但 `python -m ingest.*` / `uvicorn app.main:app` 需从 `backend/` 目录执行。
