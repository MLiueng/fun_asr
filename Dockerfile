# ============================================================
# Fun-ASR Server — GPU 模式，最小化镜像
# ============================================================
# 基于 CUDA 12.1 runtime 镜像（CUDA 12.1 ≈ 驱动 ≥ 525，兼容 Tesla V100）
# 比 pytorch 官方镜像小约 2GB（无需预装无用 torch/torchvision）
ARG BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

# Python 生产环境优化：不生成 .pyc 文件 + 不缓冲输出
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 系统依赖（APT 层，下载快，构建缓存友好）
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python-is-python3 ffmpeg && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# PyTorch + torchaudio（cu121，系统 CUDA 12.1 完全匹配）
# 独立 RUN 层：下载失败时可利用缓存重试，不用重跑 apt
# 如国内网络慢，可取消注释 --index-url 行改为清华镜像
RUN pip install --no-cache-dir --default-timeout=300 --retries 10 \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu121
#   torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 应用依赖（包较小，下载快）
RUN pip install --no-cache-dir \
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

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/api/health')" || exit 1

ENTRYPOINT ["python", "-m", "app.main"]
CMD ["--host", "0.0.0.0", "--port", "9000"]
