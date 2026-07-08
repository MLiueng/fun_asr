# Fun-ASR Server CPU 模式 — 离线部署指南

> 适用于：无 GPU 服务器 / GPU 驱动不兼容时的备选方案  
> 流程：外网构建 → 导出镜像+模型 → 内网部署

CPU 模式不需要 NVIDIA 驱动和 GPU，任何 x86_64 Linux 服务器都可运行。推理速度约为 GPU 的 5-10 倍，适合轻量级/并发低场景。

---

## 一、外网环境：构建 & 导出

### 1.1 构建镜像并预下载模型

```bash
# 进入 cpu 目录
cd Fun-ASR-vllm/cpu

# 构建 CPU 镜像
docker compose build

# 启动容器（自动预加载模型，模型会下载到 ./models 目录）
docker compose up -d

# 等待模型加载完成，查看日志确认
docker compose logs -f fun-asr-cpu

# 看到以下日志说明成功：
#   Model 'paraformer-zh' loaded successfully
#   Uvicorn running on http://0.0.0.0:9000
```

> **注意**：模型文件存储在 `cpu/models` 目录（由 docker-compose.yml 中 volume 映射），首次启动会自动从 ModelScope 下载，约 1.2GB。
>
> **Swagger UI 离线资源**：CPU 镜像直接复制源码内的 `app/static/swagger-ui/`，不再在构建阶段访问公网 CDN；`/docs` 页面只加载本地 CSS/JS。
>
> **离线防护**：部署版通过 `entrypoint.py` 启动脚本（猴子补丁）+ 离线环境变量，阻止 FunASR/ModelScope 在内网环境尝试联网检查更新或下载模型。

### 1.2 验证服务正常

```bash
# 健康检查
curl http://localhost:9000/api/health

# 语音识别测试（约 3-10 秒完成，取决于 CPU 性能和音频长度）
curl -X POST http://localhost:9000/api/asr \
  -F "file=@test.wav"
```

### 1.3 导出镜像

```bash
# 导出 CPU 镜像
docker save fun-asr-server-cpu:latest -o fun-asr-server-cpu.tar

# 压缩（可选）
gzip fun-asr-server-cpu.tar
# 结果：fun-asr-server-cpu.tar.gz
```

### 1.4 打包部署目录

```bash
# 将 cpu 目录与模型一起打包
cp -r cpu/ /tmp/fun-asr-cpu-deploy/
cp -r models/ /tmp/fun-asr-cpu-deploy/
cd /tmp
tar czf ~/fun-asr-cpu-deploy.tar.gz fun-asr-cpu-deploy/
```

> **重要**：`cpu/` 目录中包含 `entrypoint.py` 离线启动脚本，它会通过猴子补丁拦截 FunASR/ModelScope 的联网请求。打包时务必包含此文件。

### 1.5 需要传输到内网的文件

| 文件 | 说明 | 大小（参考） |
|------|------|-------------|
| `fun-asr-server-cpu.tar.gz` | Docker 镜像（CPU 版） | ~2GB |
| `fun-asr-cpu-deploy.tar.gz` | 部署目录（含 docker-compose.yml + entrypoint.py + 模型文件） | ~1.2GB |

> **对比 GPU 版**：CPU 镜像约 2GB vs GPU 镜像约 4GB，传输更快。

---

## 二、内网环境：部署

### 2.1 前置条件

| 要求 | 说明 |
|------|------|
| 操作系统 | Linux x86_64（Ubuntu 20.04+ / CentOS 7+） |
| Docker | >= 20.10 |
| CPU | 建议 4 核以上（推理时单核也可运行，但较慢） |
| 内存 | >= 4GB 可用（模型加载约需 2GB） |
| 磁盘空间 | >= 6GB 可用 |

> **无需 GPU**，无需安装 NVIDIA 驱动或 NVIDIA Container Toolkit。

### 2.2 加载镜像

```bash
# 解压并加载
gunzip fun-asr-server-cpu.tar.gz
docker load -i fun-asr-server-cpu.tar

# 验证加载成功
docker images | grep fun-asr-server-cpu
```

### 2.3 解压部署目录

```bash
cd /opt
tar xzf fun-asr-cpu-deploy.tar.gz

# 解压后目录结构：
# /opt/fun-asr-cpu-deploy/
# ├── docker-compose.yml    # CPU 模式编排文件
# ├── entrypoint.py         # 离线启动脚本（猴子补丁拦截联网请求）
# ├── Dockerfile            # CPU 版 Dockerfile（参考用）
# ├── OFFLINE_DEPLOY.md     # 本文档
# └── models/               # 模型文件
#     └── models/
#         └── iic/
#             ├── speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/
#             ├── punc_ct-transformer_cn-en-common-vocab471067-large/
#             └── ...
```

### 2.4 启动服务

```bash
cd /opt/fun-asr-cpu-deploy

# 启动
docker compose up -d

# 查看日志
docker compose logs -f fun-asr-cpu
```

CPU 模式下模型加载时间较长（约 30-60 秒），请耐心等待日志中出现 `Uvicorn running`。

### 2.5 验证服务

```bash
# 健康检查
curl http://localhost:9000/api/health

# 预期返回：
# {"code":0,"message":"success","data":{"default_model":"paraformer-zh","loaded_models":["paraformer-zh"]}}

# Swagger UI 页面（浏览器访问）
# http://localhost:9000/docs

# 验证 Swagger UI 静态资源来自本地服务
curl -I http://localhost:9000/static/swagger-ui/swagger-ui-bundle.js
curl -I http://localhost:9000/static/swagger-ui/swagger-ui.css

# 语音识别测试（CPU 模式首次较慢，约 3-10 秒）
curl -X POST http://localhost:9000/api/asr \
  -F "file=@test.wav"
```

---

## 三、配置修改

CPU 版 docker-compose.yml 仅需修改以下内容：

### 修改端口

```bash
# 方式一：环境变量（推荐）
PORT=9010 docker compose up -d

# 方式二：编辑 docker-compose.yml
# ports:
#   - "${PORT:-9010}:9000"
```

### 关闭预加载

去掉 `--preload`，首次调用识别接口时自动加载：

```yaml
    command:
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "9000"
      - "--device"
      - "cpu"
      # 删除 "--preload" 这一行
```

### 修改模型挂载路径

```yaml
    volumes:
      - /data/fun-asr/models:/app/models   # 绝对路径
```

### 日志持久化（可选）

容器内日志文件位于 `/app/logs/`，按天自动轮转（仅保留当天 + 昨天）。如需持久化到宿主机：

```yaml
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs        # 日志持久化
```

### 配置项速查

| 配置项 | docker-compose.yml 位置 | 环境变量 | 默认值 | 说明 |
|--------|------------------------|----------|--------|------|
| 服务端口 | `ports` 冒号左侧 | `PORT` | `9000` | 宿主机监听端口 |
| 推理设备 | `command` 中 `--device` | — | `cpu` | 固定为 cpu |
| 自动预加载 | `command` 中 `--preload` | — | 有 | 删掉即关闭 |
| 模型路径 | `volumes` 左侧 | — | `./models` | 宿主机模型目录 |
| 离线模式 | `environment` | `MODELSCOPE_OFFLINE` | `1` | 阻止 ModelScope 联网 |
| 离线模式 | `environment` | `HF_HUB_OFFLINE` | `1` | 阻止 HuggingFace 联网 |
| 离线模式 | `environment` | `TRANSFORMERS_OFFLINE` | `1` | 阻止 Transformers 联网 |

---

## 四、性能参考

| 场景 | CPU (Xeon 8核) | GPU (V100) |
|------|---------------|------------|
| 模型加载 | ~30-60s | ~5-10s |
| 10s 音频识别 | ~3-5s | ~0.5s |
| 60s 音频识别 | ~15-30s | ~2-4s |
| 内存占用 | ~2GB | ~4GB(GPU)+2GB(RAM) |
| 镜像大小 | ~2GB | ~4GB |

> CPU 模式适合：测试验证、低并发场景（< 5 QPS）、无 GPU 的虚拟机。  
> 生产高并发场景建议使用 GPU 版。

---

## 五、常见问题

### Q1: 启动后一直卡在模型加载？

CPU 加载模型需要 30-60 秒，比 GPU 慢得多。观察日志：

```bash
docker compose logs -f fun-asr-cpu
```

看到 `Model 'paraformer-zh' loaded in XXs` 再继续等待 `Uvicorn running` 出现即可。

### Q2: 识别速度太慢？

CPU 推理速度受限于 CPU 核心数和主频：
- 确保未和其他进程争抢 CPU
- 可限制并发请求数（上层网关/Nginx 配置）
- 长音频可拆分为短片段分别调用

### Q3: 内网模型仍然尝试从网络下载？

部署版 `docker-compose.yml` 通过 `entrypoint.py` 离线启动脚本 + 环境变量双重拦截，正常情况不会联网。如仍有问题，按以下步骤排查：

**1. 确认 entrypoint.py 已正确挂载**

```bash
# 查看容器内 entrypoint.py 是否存在
docker exec fun-asr-server-cpu ls -la /app/entrypoint.py

# 查看启动日志中是否有补丁生效信息
docker compose logs fun-asr-cpu 2>&1 | grep entrypoint
# 预期输出：
#   [entrypoint] Set MODELSCOPE_OFFLINE=1
#   [entrypoint] Set HF_HUB_OFFLINE=1
#   [entrypoint] Set TRANSFORMERS_OFFLINE=1
#   [entrypoint] AutoModel patched: disable_update=True forced
#   [entrypoint] ModelScope snapshot_download patched: local_files_only=True forced
```

**2. 确认模型文件已完整挂载**

```bash
ls -la ./models/models/iic/
docker exec fun-asr-server-cpu ls -la /app/models/models/iic/
```

**3. 确认离线环境变量生效**

```bash
docker exec fun-asr-server-cpu env | grep -E "OFFLINE|MODELSCOPE_CACHE"
# 预期输出：
#   MODELSCOPE_OFFLINE=1
#   HF_HUB_OFFLINE=1
#   TRANSFORMERS_OFFLINE=1
#   MODELSCOPE_CACHE=/app/models
#   MODELSCOPE_MODULES_CACHE=/app/models
```

### Q4: 内存不足？

模型加载约需 2GB 内存，可通过 Docker 限制：

```yaml
    # docker-compose.yml 中添加
    mem_limit: 4g
```

### Q5: CPU 版和 GPU 版模型文件是否通用？

**完全通用**。CPU 版和 GPU 版使用相同的 `./models` 目录格式，模型文件无需区分。可以共用同一份模型文件。
