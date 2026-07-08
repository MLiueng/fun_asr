# Fun-ASR Server

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green?logo=apache&logoColor=white)

**开箱即用的中文语音识别（ASR）服务 · GPU/CPU 双模 · 内网离线部署**

[快速开始](#快速开始) · [API 文档](#api-接口) · [部署指南](#部署) · [常见问题](#常见问题)

</div>

---

## 项目简介

Fun-ASR Server 是一个基于 [FunASR](https://github.com/modelscope/FunASR) 模型库和 [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) 中文语音识别模型的 **RESTful API 服务**。

它将 FunASR 的模型推理能力封装为标准的 HTTP 接口，提供文件上传识别和 URL 识别两种方式，自带 VAD 语音分段和标点恢复，开箱即用，无需编写任何推理代码。

### 核心能力

- **中文语音识别** — 基于 Paraformer-large 模型，工业级识别精度
- **VAD 自动分段** — 内置 fsmn-vad，支持任意长度音频（最长 60s/段）
- **标点恢复** — 内置 ct-punc，自动添加逗号、句号等标点
- **多格式支持** — wav / mp3 / flac / m4a / ogg / aac / wma / opus（ffmpeg 自动转码）
- **热词增强** — 支持传入热词列表提升专有名词识别率
- **文本后处理** — 自动去除 ASR 常见的重复短语（如"去去去那个"）

---

## 为什么不用 FunASR 官方服务？

> **TL;DR**：FunASR 官方提供的 [`runtime`](https://github.com/modelscope/FunASR/tree/main/runtime) 服务代码存在多个无法运行的问题，在生产环境中反复踩坑后，本项目选择直接调用 FunASR 的 **Python 模型库**（`funasr` 包）来构建可靠的服务层。

### FunASR 官方服务存在的问题

| 问题 | 详情 |
|------|------|
| **启动即崩溃** | 官方 `runtime` 服务依赖 grpc + asyncio，在不同环境中经常出现 `ImportError`、`grpc` 版本冲突，无法正常启动 |
| **文档与代码不同步** | README 中的部署步骤与实际代码逻辑不一致，按文档操作常常报错 |
| **联网行为无法关闭** | 官方服务在启动时会强制检查更新、拉取模型元信息，内网环境直接卡死或超时 |
| **模型管理缺失** | 没有提供模型加载/卸载 API，无法在运行时动态管理模型生命周期 |
| **错误处理粗糙** | 遇到异常音频或超长音频时，服务直接崩溃而非返回错误信息 |

### 本项目的解决思路

本项目 **不使用 FunASR 的服务层**，而是：

```
传统方案:  你的应用 → FunASR runtime 服务（grpc, 问题多）→ 模型
本项目:    你的应用 → FunASR Server（FastAPI RESTful, 稳定可靠）→ funasr 库 → 模型
```

| 对比 | FunASR 官方 runtime | **Fun-ASR Server（本项目）** |
|------|---------------------|------|
| 服务框架 | grpc + 自研服务 | **FastAPI + Uvicorn（成熟稳定）** |
| API 风格 | protobuf grpc | **标准 RESTful JSON** |
| 模型管理 | 无 | **完整的加载/卸载/状态查询 API** |
| 内网离线 | 无法关闭联网 | **三层拦截：环境变量 + 猴子补丁 + 本地静态资源** |
| 错误处理 | 崩溃 | **统一错误码 + 异常捕获 + 日志** |
| 部署 | 复杂、文档不同步 | **Docker 一键部署 + 详细文档** |
| 音频预处理 | 需自行处理 | **内置 ffmpeg 自动转码 16kHz mono** |
| 文本后处理 | 无 | **自动去重 + 热词支持** |

> **注意**：本项目仍然使用 FunASR 的 **模型和推理库**（`funasr` Python 包），只是不使用它有问题的 **服务层代码**。模型本身（Paraformer-large）是达摩院开源的高质量中文 ASR 模型，完全可靠。

---

## 特性

- **GPU/CPU 双模式**
  - GPU：`nvidia/cuda:12.1.1-runtime` 基础镜像，兼容驱动 ≥ 525（含 Tesla V100）
  - CPU：`python:3.11-slim` 轻量镜像，无 GPU 依赖，体积仅 ~2GB
- **内网离线部署**
  - Swagger UI 静态文件随源码打包，`/docs` 只加载本地 CSS/JS，不依赖公网 CDN
  - `entrypoint.py` 启动脚本通过猴子补丁拦截 FunASR/ModelScope 的联网请求
  - 三层离线防护确保内网环境零外网依赖
- **日志轮转**
  - 按天自动切分，仅保留当天日志 + 1 份历史备份，避免磁盘占满
- **模型缓存持久化**
  - 模型文件挂载宿主机，首次下载后重启容器不重下
- **可扩展架构**
  - 抽象基类 + 注册表模式，添加新模型只需实现 4 个方法 + 注册一行

---

## 架构设计

```
                         ┌─────────────────────────────────┐
                         │          FastAPI (9000)          │
                         │  ┌───────────┐  ┌─────────────┐  │
   HTTP Request ────────►│  │  Routes   │─►│  Schemas    │  │
                         │  └─────┬─────┘  └─────────────┘  │
                         │        │                          │
                         │  ┌─────▼─────────────────────┐   │
                         │  │     ModelManager          │   │
                         │  │  (注册表 + 生命周期管理)    │   │
                         │  │                           │   │
                         │  │  ┌─────────────────────┐  │   │
                         │  │  │  ParaformerZHModel  │  │   │
                         │  │  │  ┌────────────────┐  │  │   │
                         │  │  │  │  funasr.AutoModel │  │  │   │
                         │  │  │  │  ├─ ASR (paraformer) │  │   │
                         │  │  │  │  ├─ VAD (fsmn-vad)  │  │   │
                         │  │  │  │  └─ Punc (ct-punc)  │  │   │
                         │  │  │  └────────────────┘  │  │   │
                         │  │  └─────────────────────┘  │   │
                         │  └───────────────────────────┘   │
                         └─────────────────────────────────┘
```

### 请求处理流程

```
1. 客户端 POST /api/asr（上传音频文件）
2. FastAPI 路由接收 → 保存到临时文件
3. ModelManager 分发到 ParaformerZHModel
4. ffmpeg 预处理：转码 → 16kHz mono WAV
5. funasr.AutoModel.generate() 推理
6. 文本后处理：去重复短语
7. 返回 JSON: { code, message, data: { text, model, ... } }
8. 清理临时文件
```

---

## 项目结构

```
fun_asr/
├── app/                            # 应用主代码
│   ├── __init__.py
│   ├── main.py                     # FastAPI 入口 + CLI + 日志配置
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py               # API 路由（模型管理 + 语音识别 + 健康检查）
│   │   └── schemas.py              # 请求/响应 Pydantic 模型
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py               # 全局配置（环境变量驱动）
│   │   └── manager.py              # 模型管理器（注册表 + 生命周期 + 分发）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                 # ASR 模型抽象基类 + 音频转码 + 文本去重
│   │   └── paraformer.py           # paraformer-zh 模型实现
│   └── static/
│       └── swagger-ui/             # 本地 Swagger UI CSS/JS（离线 /docs）
├── docker/                         # GPU 离线部署文件
│   ├── docker-compose.yml          # 部署版编排（纯 image，无 build）
│   ├── entrypoint.py               # 离线启动脚本（猴子补丁拦截联网）
│   └── OFFLINE_DEPLOY.md           # GPU 离线部署完整指南
├── cpu/                            # CPU 模式
│   ├── Dockerfile                  # CPU 专用 Dockerfile
│   ├── docker-compose.yml          # CPU 编排 + 构建配置
│   ├── entrypoint.py               # 离线启动脚本（同 GPU 版）
│   └── OFFLINE_DEPLOY.md           # CPU 离线部署完整指南
├── docker-compose.yml              # GPU 开发版（含 build）
├── Dockerfile                      # GPU 专用 Dockerfile
├── requirements.txt
├── LICENSE                         # Apache License 2.0
└── README.md
```

---

## 已支持模型

| 模型 | 标识 | 说明 |
|------|------|------|
| **Paraformer-zh** | `paraformer-zh` | 达摩院开源中文语音识别模型，自带 VAD + 标点恢复 |

### 模型组成

| 子模型 | 来源 | 功能 |
|--------|------|------|
| ASR 主模型 | `paraformer-zh` (iic) | 中文语音 → 文字 |
| VAD | `fsmn-vad` (iic) | 语音活动检测，自动切分长音频 |
| 标点 | `ct-punc` (iic) | 标点恢复，自动添加逗号/句号等 |

> 模型首次运行时从 ModelScope 自动下载（~1.2GB），缓存在 `./models` 目录，后续启动直接加载本地缓存。

---

## 快速开始

### 方式一：Docker 部署（推荐）

#### GPU 模式

```bash
# 克隆项目
git clone https://github.com/your-username/fun-asr.git
cd fun-asr

# 构建并启动（首次会自动下载模型，约 1.2GB）
docker compose up -d --build

# 查看启动日志
docker compose logs -f fun-asr

# 等待看到以下日志即表示就绪：
#   [paraformer-zh] Model loaded in XXs.
#   Uvicorn running on http://0.0.0.0:9000
```

#### CPU 模式

```bash
cd cpu/
docker compose up -d --build

# 查看日志
docker compose logs -f fun-asr-cpu
```

### 方式二：本地运行

```bash
# 安装依赖（需 Python 3.10+，PyTorch 需自行安装匹配 CUDA 版本）
pip install -r requirements.txt

# 启动服务，预加载模型
python -m app.main --preload --port 9000
```

### 测试识别

```bash
# 健康检查
curl http://localhost:9000/api/health

# 文件上传识别
curl -X POST http://localhost:9000/api/asr \
  -F "file=@your-audio.wav"

# URL 方式识别
curl -X POST http://localhost:9000/api/asr/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/audio.wav"}'

# 浏览器打开 Swagger UI 交互文档
# http://localhost:9000/docs
```

### 响应示例

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "text": "今天天气真好，我们去公园散步吧。",
    "model": "paraformer-zh",
    "filename": "test.wav",
    "file_size": 320000
  }
}
```

---

## 部署

本项目提供三种部署模式，详见对应的部署指南：

| 模式 | 目录 | 适用场景 | 指南 |
|------|------|----------|------|
| **GPU 开发** | 项目根目录 | 本地开发、测试 | 本文档 |
| **GPU 离线** | `docker/` | 生产内网部署 | [docker/OFFLINE_DEPLOY.md](docker/OFFLINE_DEPLOY.md) |
| **CPU 离线** | `cpu/` | 无 GPU 服务器 | [cpu/OFFLINE_DEPLOY.md](cpu/OFFLINE_DEPLOY.md) |

### 系统要求

#### GPU 模式

| 要求 | 最低版本 |
|------|----------|
| OS | Linux（Ubuntu 20.04+ / CentOS 7+） |
| Docker | ≥ 20.10 |
| NVIDIA Container Toolkit | 已安装 |
| GPU 驱动 | ≥ 525（CUDA 12.1） |
| GPU 显存 | ≥ 4GB（V100 / T4 / A10 / 3090 等） |
| 磁盘空间 | ≥ 10GB |

#### CPU 模式

| 要求 | 最低版本 |
|------|----------|
| OS | Linux x86_64（Ubuntu 20.04+ / CentOS 7+） |
| Docker | ≥ 20.10 |
| CPU | 建议 4 核+ |
| 内存 | ≥ 4GB |
| 磁盘空间 | ≥ 6GB |

### 性能参考

| 场景 | CPU (Xeon 8核) | GPU (V100) |
|------|:--------------:|:----------:|
| 模型加载 | ~30-60s | ~5-10s |
| 10s 音频识别 | ~3-5s | ~0.5s |
| 60s 音频识别 | ~15-30s | ~2-4s |
| 内存占用 | ~2GB | ~4GB VRAM + 2GB RAM |
| 镜像大小 | ~2GB | ~4GB |

> CPU 模式适合测试验证和低并发场景（< 5 QPS）；生产高并发场景建议使用 GPU 版。

---

## API 接口

### API 文档（Swagger UI）

```
GET /docs
```

`/docs` 页面使用镜像内置的本地 Swagger UI 静态资源，不访问公网 CDN，内网环境完全可用。

---

### 模型管理

#### 查询模型状态

```http
GET /api/models/status
```

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "default_model": "paraformer-zh",
    "models": [
      { "model_name": "paraformer-zh", "loaded": true }
    ]
  }
}
```

#### 加载模型

```http
POST /api/models/load?model_name=paraformer-zh
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | 否 | 模型名称，留空使用默认 `paraformer-zh` |

#### 卸载模型

```http
DELETE /api/models/unload?model_name=paraformer-zh
```

---

### 语音识别

#### 文件上传识别

```http
POST /api/asr
Content-Type: multipart/form-data
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 音频文件，支持 wav/mp3/flac/m4a/ogg 等格式 |

> 模型会自动加载（若尚未加载）。非 WAV 格式自动通过 ffmpeg 转码为 16kHz 单声道。

#### URL 方式识别

```http
POST /api/asr/url
Content-Type: application/json
```

```json
{
  "url": "https://example.com/audio.wav"
}
```

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 语音文件的 URL 地址（服务端会下载） |

#### 识别响应

```json
{
  "code": 0,
  "message": "success",
  "data": {
    "text": "识别出的文字内容",
    "model": "paraformer-zh",
    "filename": "audio.wav",
    "file_size": 320000
  }
}
```

---

### 系统

#### 健康检查

```http
GET /api/health
```

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "default_model": "paraformer-zh",
    "loaded_models": ["paraformer-zh"]
  }
}
```

---

## 配置

### CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `9000` | 监听端口 |
| `--device` | `cuda:0` | 推理设备（CPU 模式用 `cpu`） |
| `--preload` | 关闭 | 启动时预加载默认模型 |

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `9000` | 服务端口 |
| `DEVICE` | `cuda:0` | 推理设备 |
| `DEFAULT_MODEL` | `paraformer-zh` | 默认模型 |
| `PARAFORMER_MODEL_ID` | `paraformer-zh` | Paraformer 模型 ID（支持本地路径） |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `LOG_DIR` | `/app/logs` | 日志文件目录（容器内） |

### 离线部署环境变量

> 仅部署版 `docker/docker-compose.yml` 和 `cpu/docker-compose.yml` 中设置，开发版无需配置。

| 变量 | 值 | 说明 |
|------|----|------|
| `MODELSCOPE_OFFLINE` | `1` | 阻止 ModelScope 联网下载 |
| `HF_HUB_OFFLINE` | `1` | 阻止 HuggingFace 联网下载 |
| `TRANSFORMERS_OFFLINE` | `1` | 阻止 Transformers 联网下载 |

部署版还通过 `entrypoint.py` 猴子补丁进一步拦截：
- `funasr.AutoModel.__init__` 强制 `disable_update=True`（阻止检查更新）
- `modelscope.hub.snapshot_download` 强制 `local_files_only=True`（阻止下载模型）

---

## 添加新模型

项目采用 **抽象基类 + 注册表** 模式，扩展新模型只需 2 步：

### 1. 实现模型类

```python
# app/models/my_model.py
from .base import BaseASRModel

class MyModel(BaseASRModel):
    @property
    def name(self) -> str:
        return "my-model"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        # 加载模型权重
        ...

    def unload(self) -> None:
        # 释放模型资源
        ...

    def transcribe(self, audio_path: str, **kwargs) -> str:
        # 执行语音识别，返回识别文本
        ...
```

### 2. 注册到管理器

```python
# app/core/manager.py
from ..models.my_model import MyModel

_MODEL_REGISTRY: Dict[str, Type[BaseASRModel]] = {
    "paraformer-zh": ParaformerZHModel,
    "my-model": MyModel,           # 新增一行
}
```

注册后，新模型自动获得完整的 API 支持：
- `POST /api/models/load?model_name=my-model`
- `DELETE /api/models/unload?model_name=my-model`
- `POST /api/asr` （通过 `model_name` 参数指定）

---

## 离线防护机制

内网环境无法访问公网，而 FunASR/ModelScope 默认会尝试联网检查更新和下载模型，导致服务启动失败或超时。部署版通过 **三层防护** 彻底阻断：

| 层级 | 机制 | 拦截目标 |
|------|------|----------|
| **1. 环境变量** | `MODELSCOPE_OFFLINE=1` 等 | 告知各库进入离线模式 |
| **2. AutoModel 补丁** | 猴子补丁强制 `disable_update=True` | 阻止 "Check update of funasr" |
| **3. ModelScope 补丁** | 猴子补丁强制 `local_files_only=True` | 阻止 "download models from model hub" |

> `entrypoint.py` 在 `app.main` 启动前，**同进程内**（关键：subprocess 会使补丁失效）对 `funasr.AutoModel` 和 `modelscope.hub.snapshot_download` 进行猴子补丁，覆盖其网络请求行为。

---

## 开发版 vs 部署版对比

| 项目 | 开发版（根目录） | 部署版（`docker/` 或 `cpu/`） |
|------|-----------------|-------------------------------|
| `build` | 有，从 Dockerfile 构建 | **无**，直接用 `image` |
| `entrypoint` | 默认（`python -m app.main`） | `python /app/entrypoint.py`（离线补丁） |
| `volumes` | `./models` | `./models` + `./entrypoint.py:ro` |
| 离线环境变量 | 无 | `MODELSCOPE_OFFLINE=1` 等 |

---

## 常见问题

<details>
<summary><b>Q: 报错 "CUDA error: no kernel image is available"？</b></summary>

GPU 驱动版本与镜像 CUDA 版本不匹配。当前镜像使用 CUDA 12.1（驱动 ≥ 525）。

**解决方案：**
1. **CPU 模式** — 使用 `cpu/` 目录部署
2. **重建兼容镜像** — 修改 Dockerfile 中基础镜像和 torch 下载地址：
   - 驱动 ≥ 520 → `nvidia/cuda:11.8.0` + `torch --index-url .../cu118`
   - 驱动 ≥ 550 → `nvidia/cuda:12.4.0` + `torch --index-url .../cu124`
</details>

<details>
<summary><b>Q: 内网部署时模型仍然尝试联网下载？</b></summary>

确认 `entrypoint.py` 已正确挂载：

```bash
# 检查容器内文件
docker exec fun-asr-server ls -la /app/entrypoint.py

# 检查启动日志中的补丁信息
docker compose logs fun-asr 2>&1 | grep entrypoint
```

预期输出包含 `AutoModel patched` 和 `snapshot_download patched`。
</details>

<details>
<summary><b>Q: nvidia-smi 可用但容器内看不到 GPU？</b></summary>

安装 NVIDIA Container Toolkit：

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
</details>

<details>
<summary><b>Q: 如何修改监听端口？</b></summary>

```bash
# 环境变量方式（推荐）
PORT=9000 docker compose up -d
```
</details>

<details>
<summary><b>Q: CPU 版和 GPU 版模型文件是否通用？</b></summary>

**完全通用**。CPU 版和 GPU 版使用相同的 `./models` 目录格式，模型文件无需区分。
</details>

更多问题详见：
- [GPU 离线部署指南 - 常见问题](docker/OFFLINE_DEPLOY.md)
- [CPU 离线部署指南 - 常见问题](cpu/OFFLINE_DEPLOY.md)

---

## 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | [FastAPI](https://fastapi.tiangolo.com/) | 高性能异步 Web 框架 |
| ASGI 服务器 | [Uvicorn](https://www.uvicorn.org/) | 基于 uvloop 的高性能服务器 |
| 模型库 | [FunASR](https://github.com/modelscope/FunASR) | 达摩院语音识别模型库 |
| ASR 模型 | [Paraformer-large](https://arxiv.org/abs/2206.08317) | 工业级中文语音识别 |
| 深度学习 | [PyTorch](https://pytorch.org/) | GPU: cu121 / CPU |
| 音频处理 | [ffmpeg](https://ffmpeg.org/) | 音频格式转码 |
| 容器化 | [Docker](https://www.docker.com/) | GPU + CPU 双模镜像 |

---

## 开发

```bash
# 克隆项目
git clone https://github.com/your-username/fun-asr.git
cd fun-asr

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 本地启动（开发模式，DEBUG 日志）
LOG_LEVEL=DEBUG python -m app.main --preload --port 9000
```

### 代码结构说明

- `app/models/base.py` — 定义 `BaseASRModel` 抽象基类，以及 `convert_audio_to_wav`（ffmpeg 转码）和 `remove_repeated_text`（文本去重）两个工具函数
- `app/core/manager.py` — `ModelManager` 管理模型实例的创建、加载、卸载和识别分发，通过 `_MODEL_REGISTRY` 注册表实现解耦
- `app/api/routes.py` — 所有 HTTP 路由，包含模型管理（load/unload/status）、语音识别（file/url）、健康检查
- `app/main.py` — FastAPI 应用工厂，日志配置（控制台 + 按天轮转文件），CLI 参数解析

---

## 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'Add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。

### 致谢

- [FunASR](https://github.com/modelscope/FunASR) — 达摩院语音识别模型库
- [ModelScope](https://www.modelscope.cn/) — 模型托管平台
- [Paraformer](https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) — 中文语音识别模型
- [FastAPI](https://fastapi.tiangolo.com/) — 现代 Python Web 框架

> **声明**：本项目使用 FunASR 的模型和推理库，但**不使用** FunASR 的服务层代码。本项目独立实现了基于 FastAPI 的 RESTful 服务，解决了官方服务在内网部署、稳定性、错误处理等方面的不足。
