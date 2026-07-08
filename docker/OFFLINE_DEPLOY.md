# Fun-ASR Server 离线部署指南

> 适用于：外网构建 → 导出镜像+模型 → 内网部署

---

## 一、外网环境：构建 & 导出

### 1.1 构建镜像并预下载模型

```bash
# 进入项目目录
cd Fun-ASR-vllm

# 构建镜像
docker compose build

# 启动容器（自动预加载模型，模型会下载到 ./models 目录）
docker compose up -d

# 等待模型下载完成，查看日志确认
docker compose logs -f fun-asr

# 看到以下日志说明模型加载成功：
#   Model 'paraformer-zh' loaded successfully
#   Uvicorn running on http://0.0.0.0:9000
```

> **注意**：模型文件存储在 `./models` 目录（由 docker-compose.yml 中 volume 映射），首次启动会自动从 ModelScope 下载，约 1.2GB。
>
> **Swagger UI 离线资源**：`/docs` 使用镜像内的 `app/static/swagger-ui/` 本地 CSS/JS，构建和运行时不会依赖 `cdn.jsdelivr.net`、`unpkg.com` 等公网 CDN。

### 1.2 验证服务正常

```bash
# 健康检查
curl http://localhost:9000/api/health

# 语音识别测试
curl -X POST http://localhost:9000/api/asr \
  -F "file=@test.wav"
```

### 1.3 导出镜像

```bash
# 导出镜像为 tar 文件
docker save fun-asr-server:latest -o fun-asr-server.tar

# 压缩（可选，可减小约 40% 体积）
gzip fun-asr-server.tar
# 结果：fun-asr-server.tar.gz
```

### 1.4 打包部署目录

部署所需文件已包含在 `docker/` 目录中，将其与模型一起打包：

```bash
# 拷贝 docker 目录到临时位置
cp -r docker/ /tmp/fun-asr-deploy/

# 将模型目录拷入
cp -r models/ /tmp/fun-asr-deploy/

# 打包整个部署目录（输出到当前用户目录）
cd /tmp
tar czf ~/fun-asr-deploy.tar.gz fun-asr-deploy/
```

> **重要**：`docker/` 目录中包含 `entrypoint.py` 离线启动脚本，它会通过猴子补丁拦截 FunASR/ModelScope 的联网请求。打包时务必包含此文件。

### 1.5 需要传输到内网的文件

| 文件 | 说明 | 大小（参考） |
|------|------|-------------|
| `fun-asr-server.tar.gz` | Docker 镜像 | ~4GB |
| `fun-asr-deploy.tar.gz` | 部署目录（含 docker-compose.yml + entrypoint.py + 模型文件） | ~1.2GB |

> 可通过 U 盘、内网文件共享等方式传输到内网服务器。

---

## 二、内网环境：部署

### 2.1 前置条件

| 要求 | 说明 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04+ / CentOS 7+） |
| Docker | >= 20.10 |
| NVIDIA Container Toolkit | 已安装并配置（GPU 模式必需） |
| GPU | NVIDIA GPU，驱动版本 >= 525.60.13（CUDA 12.x） |
| 磁盘空间 | >= 10GB 可用 |

### 2.2 验证 GPU 与驱动版本

```bash
# 1. 检查 NVIDIA 驱动版本（关注右上角 Driver Version）
nvidia-smi

# 2. 仅输出驱动版本号
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# 3. 脚本检测：驱动版本是否满足 CUDA 12.1 的最低要求（≥ 525）
driver_ver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d'.' -f1)
if [ "$driver_ver" -ge 525 ]; then
    echo "OK: 驱动版本满足要求（≥ 525，支持 CUDA 12.1）"
else
    echo "ERROR: 驱动版本不满足要求，当前 $driver_ver，需要 ≥ 525"
fi

# 4. 检查 Docker 能否调用 GPU
docker run --rm --gpus all nvidia/cuda:12.1.1-runtime-ubuntu22.04 nvidia-smi
```

> **驱动兼容性速查**：CUDA 12.1 → 驱动 ≥ 525（当前镜像目标）。如果驱动版本不足，可改用 CPU 模式部署（参见 `cpu/OFFLINE_DEPLOY.md`）。

### 2.3 加载镜像

```bash
# 如果是 .tar.gz
gunzip fun-asr-server.tar.gz
docker load -i fun-asr-server.tar

# 如果是 .tar
docker load -i fun-asr-server.tar

# 验证镜像已加载
docker images | grep fun-asr
```

### 2.4 解压部署目录

```bash
# 解压到 /opt
cd /opt
tar xzf fun-asr-deploy.tar.gz

# 解压后目录结构：
# /opt/fun-asr-deploy/
# ├── docker-compose.yml    # 编排文件（已配置离线部署）
# ├── entrypoint.py         # 离线启动脚本（猴子补丁拦截联网请求）
# └── models/               # 模型文件
#     └── models/
#         └── iic/
#             ├── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/  # ASR 主模型
#             ├── punc_ct-transformer_cn-en-common-vocab471067-large/                    # 标点恢复模型
#             └── ...
```

### 2.5 验证镜像 CUDA/Torch 版本

启动前先确认镜像内部的 CUDA 和 PyTorch 版本与预期一致：

```bash
# 1. 查看镜像内的 CUDA 版本
docker run --rm fun-asr-server:latest nvidia-smi

# 2. 查看镜像内的系统 CUDA 版本信息
docker run --rm fun-asr-server:latest cat /usr/local/cuda/version.json

# 3. 查看 PyTorch 编译时的 CUDA 版本（需覆盖默认入口点）
docker run --rm --entrypoint python fun-asr-server:latest -c "import torch; import torchaudio; print(torch.version.cuda, torchaudio.__version__)"

# 预期输出（当前镜像策略）：
#   12.1 2.6.x
```

> 若 `torch.version.cuda` 不是 `12.1`，说明镜像构建时 torch 下载了错误的 CUDA 版本，需检查 Dockerfile 中的 `--index-url`。

### 2.6 启动服务

```bash
cd /opt/fun-asr-deploy

# 启动
docker compose up -d

# 查看日志，确认模型加载成功
docker compose logs -f fun-asr
```

### 2.7 验证服务

```bash
# 健康检查
curl http://localhost:9000/api/health

# 预期返回：
# {"code":0,"message":"success","data":{"default_model":"paraformer-zh","loaded_models":["paraformer-zh"]}}

# 模型状态
curl http://localhost:9000/api/models/status

# 语音识别（上传文件）
curl -X POST http://localhost:9000/api/asr \
  -F "file=@test.wav"

# Swagger UI 页面（浏览器访问）
# http://localhost:9000/docs

# 验证 Swagger UI 静态资源来自本地服务
curl -I http://localhost:9000/static/swagger-ui/swagger-ui-bundle.js
curl -I http://localhost:9000/static/swagger-ui/swagger-ui.css

# 语音识别（URL 方式，需内网可访问的 URL）
curl -X POST http://localhost:9000/api/asr/url \
  -H "Content-Type: application/json" \
  -d '{"url": "http://internal-server/audio/test.wav"}'
```

---

## 三、docker-compose.yml 说明

部署目录中的 `docker-compose.yml` 与开发版的关键差异：

| 项目 | 开发版（项目根目录） | 部署版（docker/） |
|------|---------------------|-------------------|
| `build` | 有，从 Dockerfile 构建 | **无**，直接用 `image` |
| `entrypoint` | 默认（`python -m app.main`） | `python /app/entrypoint.py`（离线补丁） |
| `volumes` | `./models:/app/models` | `./models:/app/models` + `./entrypoint.py:/app/entrypoint.py:ro` |
| 离线环境变量 | 无 | `MODELSCOPE_OFFLINE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |

### 离线防护机制

内网环境无法访问公网，FunASR/ModelScope 默认会尝试联网检查更新和下载模型，导致服务启动失败。部署版通过三层防护彻底阻断：

| 层级 | 机制 | 作用 |
|------|------|------|
| 环境变量 | `MODELSCOPE_OFFLINE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` | 告知各库进入离线模式，不尝试网络请求 |
| AutoModel 补丁 | `entrypoint.py` 猴子补丁强制 `disable_update=True` | 阻止 "Check update of funasr" |
| ModelScope 补丁 | `entrypoint.py` 猴子补丁强制 `snapshot_download(local_files_only=True)` | 阻止 "download models from model hub: ms" |

> **原理**：`entrypoint.py` 在 `app.main` 启动前，同进程内对 `funasr.AutoModel` 和 `modelscope.hub.snapshot_download` 进行猴子补丁（monkey-patch），覆盖其网络请求行为。因为是同进程调用（`from app.main import main; main()`），补丁对实际服务完全生效。

---

## 四、配置修改

内网部署时可通过 **环境变量** 或 **直接编辑 docker-compose.yml** 两种方式修改配置。

### 方式一：启动时传环境变量（推荐，不改文件）

```bash
# 修改端口号为 9010，使用第 2 块 GPU
PORT=9010 GPU_ID=1 docker compose up -d

# 使用 CPU 模式（需同时修改 docker-compose.yml，见下方说明）
PORT=9000 docker compose up -d
```

### 方式二：编辑 docker-compose.yml（永久生效）

```bash
vi docker-compose.yml
```

以下为各配置项的修改示例：

#### 1. 修改服务端口

默认 `9000`，改为 `9010`：

```yaml
    ports:
      - "${PORT:-9010}:9000"   # 宿主机端口:容器端口
```

> 只改冒号左边的宿主机端口即可，容器内端口 `9000` 不需要改。

#### 2. 指定使用的 GPU

默认使用第 `0` 号 GPU，改为第 `1` 号：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]    # GPU 编号，从 0 开始
              capabilities: [gpu]
```

使用多块 GPU：

```yaml
              device_ids: ["0", "1"]   # 同时使用 GPU 0 和 GPU 1
```

> 通过 `nvidia-smi` 查看服务器上的 GPU 编号。

#### 3. 切换为 CPU 模式

如果服务器没有 GPU，需移除 GPU 相关配置：

```yaml
services:
  fun-asr:
    image: fun-asr-server:latest
    container_name: fun-asr-server
    ports:
      - "${PORT:-9000}:9000"
    environment:
      - MODELSCOPE_CACHE=/app/models
      - MODELSCOPE_MODULES_CACHE=/app/models
    # 删除整个 deploy 段
    volumes:
      - ./models:/app/models
    command:
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "9000"
      - "--device"
      - "cpu"          # 改为 cpu
      - "--preload"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9000/api/health')"]
      interval: 30s
      timeout: 10s
      start_period: 60s
      retries: 3
```

> **注意**：CPU 模式推理速度较慢，不建议生产环境使用。

#### 4. 修改模型挂载路径

如果模型文件不在 `./models` 目录，修改 volumes：

```yaml
    volumes:
      - /data/fun-asr/models:/app/models   # 绝对路径
```

#### 5. 日志持久化（可选）

容器内日志文件位于 `/app/logs/`，按天自动轮转（仅保留当天 + 昨天）。如需持久化到宿主机：

```yaml
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs        # 日志持久化
```

#### 6. 关闭自动预加载

去掉 `--preload` 参数，服务启动后不立即加载模型，首次调用识别接口时自动加载：

```yaml
    command:
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "9000"
      - "--device"
      - "cuda:0"
      # 删除 "- "--preload"" 这行
```

### 配置项速查表

| 配置项 | docker-compose.yml 位置 | 环境变量 | 默认值 | 说明 |
|--------|------------------------|----------|--------|------|
| 服务端口 | `ports` 冒号左侧 | `PORT` | `9000` | 宿主机监听端口 |
| GPU 编号 | `deploy.resources.reservations.devices.device_ids` | `GPU_ID` | `0` | NVIDIA GPU 编号 |
| 推理设备 | `command` 中 `--device` | — | `cuda:0` | `cuda:0` / `cpu` |
| 自动预加载 | `command` 中 `--preload` | — | 有 | 删掉即关闭 |
| 模型路径 | `volumes` 左侧 | — | `./models` | 宿主机模型目录 |
| 离线模式 | `environment` | `MODELSCOPE_OFFLINE` | `1` | 阻止 ModelScope 联网 |
| 离线模式 | `environment` | `HF_HUB_OFFLINE` | `1` | 阻止 HuggingFace 联网 |
| 离线模式 | `environment` | `TRANSFORMERS_OFFLINE` | `1` | 阻止 Transformers 联网 |

---

## 五、常见问题

### Q0: 报错 "CUDA error: no kernel image is available for execution"？

当前镜像构建策略：基础镜像 `nvidia/cuda:12.1.1-runtime-ubuntu22.04`，系统 CUDA 12.1 与 PyTorch cu121 完全匹配（兼容 Tesla V100 + 驱动 535）。相比 pytorch 官方镜像节省约 2GB。

**诊断：**
```bash
nvidia-smi   # 关注右上角的 CUDA Version 和 Driver Version
# 驱动 ≥ 525 即可运行 CUDA 12.1 的 PyTorch
# Tesla V100 + 驱动 535.54.03 → 完全兼容
```

**解决方案：**

1. **CPU 模式** — 去掉 GPU，改为纯 CPU 推理（仅用于验证）：
   ```yaml
   # docker-compose.yml 修改：
   # 1. 删除 deploy 段中 device_ids 部分
   # 2. command 中 --device cuda:0 改为 --device cpu
   ```
2. **重建兼容镜像** — 在外网修改 Dockerfile 适配目标 GPU：
   ```dockerfile
   # Dockerfile 中修改基础镜像和 CUDA 版本，例如：
   # 驱动 ≥ 520 → 改 ARG BASE_IMAGE=nvidia/cuda:11.8.0-runtime-ubuntu22.04
   #            并改 torch 下载地址为 https://download.pytorch.org/whl/cu118
   # 驱动 ≥ 550 → 改 ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04
   #            并改 torch 下载地址为 https://download.pytorch.org/whl/cu124
   ```
   > 驱动版本检测：`nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1`

### Q1: 内网部署时模型仍然尝试从网络下载？

部署版 `docker-compose.yml` 通过 `entrypoint.py` 离线启动脚本 + 环境变量双重拦截，正常情况不会联网。如仍有问题，按以下步骤排查：

**1. 确认 entrypoint.py 已正确挂载**

```bash
# 查看容器内 entrypoint.py 是否存在
docker exec fun-asr-server ls -la /app/entrypoint.py

# 查看启动日志中是否有补丁生效信息
docker compose logs fun-asr 2>&1 | grep entrypoint
# 预期输出：
#   [entrypoint] Set MODELSCOPE_OFFLINE=1
#   [entrypoint] Set HF_HUB_OFFLINE=1
#   [entrypoint] Set TRANSFORMERS_OFFLINE=1
#   [entrypoint] AutoModel patched: disable_update=True forced
#   [entrypoint] ModelScope snapshot_download patched: local_files_only=True forced
```

如果看不到 `[entrypoint]` 日志，说明 `entrypoint.py` 未被正确加载，检查：
- `docker-compose.yml` 中 `entrypoint` 是否为 `["python", "/app/entrypoint.py"]`
- `volumes` 中是否有 `- ./entrypoint.py:/app/entrypoint.py:ro`
- `entrypoint.py` 文件是否在 `docker-compose.yml` 同级目录

**2. 确认模型文件已完整挂载**

```bash
# 查看宿主机模型目录
ls -la ./models/models/iic/

# 查看容器内模型目录
docker exec fun-asr-server ls -la /app/models/models/iic/
```

ModelScope 会优先从缓存目录查找模型，缓存中存在则不会下载。

**3. 确认离线环境变量生效**

```bash
# 查看容器内环境变量
docker exec fun-asr-server env | grep -E "OFFLINE|MODELSCOPE_CACHE"
# 预期输出：
#   MODELSCOPE_OFFLINE=1
#   HF_HUB_OFFLINE=1
#   TRANSFORMERS_OFFLINE=1
#   MODELSCOPE_CACHE=/app/models
#   MODELSCOPE_MODULES_CACHE=/app/models
```

### Q2: `nvidia-smi` 可用但容器内看不到 GPU？

检查 NVIDIA Container Toolkit 是否安装：

```bash
# 安装（Ubuntu）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Q3: 如何指定使用哪块 GPU？

参见 [配置修改 → 指定 GPU](#2-指定使用的-gpu)。

### Q4: 如何修改监听端口？

参见 [配置修改 → 修改端口](#1-修改服务端口)。

### Q5: 如何确认模型文件完整性？

在外网环境首次下载模型后，检查模型目录：

```bash
ls -la ./models/models/iic/
```

确保模型目录存在且非空（如 `punc_ct-transformer_cn-en-common-vocab471067-large` 等）。
