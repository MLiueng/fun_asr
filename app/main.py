"""Fun-ASR Server 入口。"""

import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import fastapi
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.staticfiles import StaticFiles

from .api import routes
from .core.config import settings
from .core.manager import ModelManager

# Swagger UI 本地静态文件路径（内网离线部署使用）
# 静态文件随源码提交到 app/static/swagger-ui/，构建镜像时由 COPY 带入
# 本地开发时也存在，无需回退到 fastapi 内置
_SWAGGER_UI_BUILTIN = Path(__file__).resolve().parent / "static" / "swagger-ui"
_SWAGGER_UI_FASTAPI = Path(fastapi.__file__).parent / "openapi" / "docs-ui" / "swagger-ui"
_SWAGGER_UI_PATH = _SWAGGER_UI_BUILTIN if _SWAGGER_UI_BUILTIN.exists() else _SWAGGER_UI_FASTAPI

# ---------------------------------------------------------------------------
# 日志：控制台 + 文件（按天轮转，仅保留当天日志 + 昨天一份备份）
# ---------------------------------------------------------------------------
_LOG_DIR = Path(settings.LOG_DIR)
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "fun-asr.log"

_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 控制台 handler
_console = logging.StreamHandler()
_console.setFormatter(_formatter)

# 文件 handler — 每天午夜轮转，只保留 1 份历史备份（即最多当天 + 昨天两份文件）
_file = TimedRotatingFileHandler(
    str(_LOG_FILE),
    when="midnight",
    interval=1,
    backupCount=1,
    encoding="utf-8",
)
_file.setFormatter(_formatter)

# 配置根 logger（uvicorn 也会继承此配置）
_root = logging.getLogger()
_root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
_root.handlers.clear()  # 清除 basicConfig 可能遗留的默认 handler
_root.addHandler(_console)
_root.addHandler(_file)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例。"""
    app = FastAPI(
        title="Fun-ASR Server",
        version="4.0.0",
        description=(
            "## Fun-ASR 语音识别服务\n\n"
            "基于 FunASR paraformer-zh 的中文语音识别服务，自带 VAD 和标点恢复。\n\n"
            "### 快速开始\n"
            "1. 调用 `POST /api/models/load` 加载模型（或直接调用识别接口自动加载）\n"
            "2. 调用 `POST /api/asr` 上传语音文件进行识别，或 `POST /api/asr/url` 通过 URL 识别\n"
        ),
        openapi_tags=[
            {
                "name": "语音识别",
                "description": "语音文件识别相关接口，支持文件上传和 URL 两种方式",
            },
            {
                "name": "模型管理",
                "description": "ASR 模型的加载、卸载和状态查询",
            },
            {
                "name": "系统",
                "description": "系统健康检查等运维接口",
            },
        ],
        docs_url=None,
    )

    # 挂载本地 Swagger UI 静态文件
    if _SWAGGER_UI_PATH.exists():
        app.mount(
            "/static/swagger-ui",
            StaticFiles(directory=str(_SWAGGER_UI_PATH)),
            name="swagger-ui-static",
        )
    else:
        logger.warning(
            "Swagger UI static files not found at %s; "
            "the docs page may not render correctly in offline environments.",
            _SWAGGER_UI_PATH,
        )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():

        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
            swagger_css_url="/static/swagger-ui/swagger-ui.css",
            swagger_favicon_url="/static/swagger-ui/favicon-32x32.png",
        )

    app.include_router(routes.router)


    # 跨域支持
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


def main():
    parser = argparse.ArgumentParser(description="Fun-ASR Server")
    parser.add_argument("--host", type=str, default=settings.HOST, help="服务监听地址")
    parser.add_argument("--port", type=int, default=settings.PORT, help="服务端口号")
    parser.add_argument("--device", type=str, default=settings.DEVICE, help="推理设备")
    parser.add_argument("--preload", action="store_true", help="启动时预加载默认模型")
    args = parser.parse_args()

    # 初始化模型管理器
    mgr = ModelManager(device=args.device)
    routes.manager = mgr
    logger.info(f"ModelManager initialized: device={args.device}, "
                f"available_models={mgr.available_models}")

    # 预加载
    if args.preload:
        try:
            logger.info(f"Pre-loading model '{settings.DEFAULT_MODEL}' ...")
            mgr.load(settings.DEFAULT_MODEL)
        except Exception as e:
            logger.error(f"Failed to pre-load model: {e}")

    app = create_app()
    logger.info(f"Starting Fun-ASR Server on {args.host}:{args.port}, log_dir={_LOG_DIR}")
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
