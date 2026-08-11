# 金匮问渠 · Docker 镜像（Hugging Face Space / Render 等均可使用）
# 构建期执行文档解析与索引构建（BM25），运行时只读查询。
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY 规章制度/ /app/规章制度/

WORKDIR /app/backend
# 构建期生成索引（未安装向量依赖时自动降级为 BM25）
RUN python -m ingest.parse && python -m ingest.chunk && python -m ingest.embed_store

ENV OPENAI_MODEL=dsv4f
EXPOSE 7860

# OPENAI_BASE_URL / OPENAI_API_KEY 通过平台 Secrets 注入
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
