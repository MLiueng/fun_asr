# ============================================================
# Fun-ASR Server — CPU 模式，纯 CPU 推理镜像
# ============================================================
# 使用轻量 Python slim 镜像，无 CUDA 依赖，体积小（~2GB）
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Python 生产环境优化：不生成 .pyc 文件 + 不缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 系统依赖 + PyTorch CPU 版 + Python 依赖（合并为两层，减少镜像体积）
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    pip install --no-cache-dir \
    torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    "funasr>=1.2.7" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "python-multipart>=0.0.9" \
    "httpx>=0.27.0"

# 复制项目文件（含 Swagger UI 静态文件，已在源码内）
COPY app/ app/


# 模型缓存目录 + 日志目录
RUN mkdir -p /app/models /app/logs
ENV HF_HOME=/app/models \
    MODELSCOPE_CACHE=/app/models \
    MODELSCOPE_MODULES_CACHE=/app/models

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/api/health')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
CMD ["--host", "0.0.0.0", "--port", "9000"]
