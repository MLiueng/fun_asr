"""API 路由定义。"""

import logging
import os
import tempfile
import time
import traceback

import httpx
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.manager import ModelManager
from .schemas import ASRUrlRequest, ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# 由 main.py 注入
manager: ModelManager | None = None


def _get_manager() -> ModelManager:
    if manager is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return manager


def _ensure_model_loaded(mgr: ModelManager) -> None:
    """确保默认模型已加载，未加载则自动加载。"""
    name = settings.DEFAULT_MODEL
    inst = mgr.get_instance(name)
    if inst is None or not inst.is_loaded:
        logger.info(f"Auto-loading model '{name}' ...")
        try:
            mgr.load(name)
        except Exception as e:
            logger.error(f"Failed to auto-load model: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


# ---------------------------------------------------------------------------
# 模型管理
# ---------------------------------------------------------------------------

@router.get(
    "/models/status",
    summary="查询模型状态",
    tags=["模型管理"],
    response_model=ApiResponse,
)
def model_status():
    """查询当前 ASR 模型的加载状态。"""
    mgr = _get_manager()
    models_info = []
    for name in mgr.available_models:
        inst = mgr.get_instance(name)
        models_info.append({
            "model_name": name,
            "loaded": inst.is_loaded if inst else False,
        })
    return JSONResponse(content={
        "code": 0,
        "message": "success",
        "data": {
            "default_model": settings.DEFAULT_MODEL,
            "models": models_info,
        },
    })


@router.post(
    "/models/load",
    summary="加载模型",
    tags=["模型管理"],
    response_model=ApiResponse,
)
def load_model(
    model_name: str = Query(
        default="",
        description="模型名称，留空使用默认模型 paraformer-zh",
    ),
):
    """加载指定 ASR 模型到内存。

    目前仅支持 **paraformer-zh**（中文语音识别模型，自带 VAD 和标点恢复）。
    """
    mgr = _get_manager()
    name = model_name or settings.DEFAULT_MODEL
    logger.info(f"API [POST /api/models/load] model={name}")
    try:
        inst = mgr.load(name)
        return JSONResponse(content={
            "code": 0,
            "message": f"Model '{name}' loaded successfully",
            "data": {"model_name": name, "loaded": inst.is_loaded},
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load model: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.delete(
    "/models/unload",
    summary="卸载模型",
    tags=["模型管理"],
    response_model=ApiResponse,
)
def unload_model(
    model_name: str = Query(
        default="",
        description="模型名称，留空使用默认模型 paraformer-zh",
    ),
):
    """卸载指定 ASR 模型，释放 GPU/CPU 资源。"""
    mgr = _get_manager()
    name = model_name or settings.DEFAULT_MODEL
    logger.info(f"API [DELETE /api/models/unload] model={name}")
    inst = mgr.get_instance(name)
    if inst is None or not inst.is_loaded:
        raise HTTPException(status_code=404, detail=f"Model '{name}' is not loaded")
    mgr.unload(name)
    return JSONResponse(content={
        "code": 0,
        "message": f"Model '{name}' unloaded successfully",
    })


# ---------------------------------------------------------------------------
# 语音识别
# ---------------------------------------------------------------------------

@router.post(
    "/asr",
    summary="语音识别（文件上传）",
    tags=["语音识别"],
    response_model=ApiResponse,
)
def asr(
    file: UploadFile = File(..., description="语音文件，支持 wav/mp3/flac/m4a/ogg 等格式"),
):
    """通过上传语音文件进行语音识别。

    - 支持格式：wav、mp3、flac、m4a、ogg 等常见音频格式
    - 使用 paraformer-zh 模型，仅支持中文识别
    - 模型会自动加载（若尚未加载）
    """
    logger.info(f"API [POST /api/asr] filename={file.filename}, content_type={file.content_type}")

    mgr = _get_manager()
    _ensure_model_loaded(mgr)

    # 保存上传文件到临时目录
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = file.file.read()
            tmp.write(content)
            tmp_path = tmp.name
        logger.debug(f"File saved: {tmp_path}, size={len(content)} bytes")
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        t0 = time.time()
        text = mgr.transcribe(settings.DEFAULT_MODEL, tmp_path)
        elapsed = time.time() - t0
        logger.info(f"ASR completed in {elapsed:.2f}s, text: {text[:80]}")

        return JSONResponse(content={
            "code": 0,
            "message": "success",
            "data": {
                "text": text,
                "model": settings.DEFAULT_MODEL,
                "filename": file.filename,
                "file_size": len(content),
            },
        })
    except Exception as e:
        logger.error(f"ASR failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post(
    "/asr/url",
    summary="语音识别（URL 方式）",
    tags=["语音识别"],
    response_model=ApiResponse,
)
def asr_url(request: ASRUrlRequest):
    """通过语音文件 URL 进行语音识别。

    服务端会从指定 URL 下载语音文件，然后进行识别。

    - 支持格式：wav、mp3、flac、m4a、ogg 等常见音频格式
    - 使用 paraformer-zh 模型，仅支持中文识别
    - 模型会自动加载（若尚未加载）
    """
    logger.info(f"API [POST /api/asr/url] url={request.url}")

    mgr = _get_manager()
    _ensure_model_loaded(mgr)

    # 从 URL 下载语音文件
    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(request.url)
            resp.raise_for_status()
            content = resp.content
    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to download audio: HTTP {e.response.status_code}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download audio: HTTP {e.response.status_code}",
        )
    except Exception as e:
        logger.error(f"Failed to download audio: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Downloaded file is empty")

    # 推断文件后缀
    url_path = request.url.split("?")[0].split("#")[0]
    suffix = os.path.splitext(url_path)[1] or ".wav"
    if suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"):
        suffix = ".wav"

    # 保存到临时文件
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.debug(f"Audio saved: {tmp_path}, size={len(content)} bytes")
    except Exception as e:
        logger.error(f"Failed to save downloaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to save downloaded file: {str(e)}")

    filename = os.path.basename(url_path) or "audio.wav"

    try:
        t0 = time.time()
        text = mgr.transcribe(settings.DEFAULT_MODEL, tmp_path)
        elapsed = time.time() - t0
        logger.info(f"ASR completed in {elapsed:.2f}s, text: {text[:80]}")

        return JSONResponse(content={
            "code": 0,
            "message": "success",
            "data": {
                "text": text,
                "model": settings.DEFAULT_MODEL,
                "filename": filename,
                "file_size": len(content),
            },
        })
    except Exception as e:
        logger.error(f"ASR failed: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"ASR failed: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="健康检查",
    tags=["系统"],
    response_model=ApiResponse,
)
def health():
    """检查服务健康状态，返回默认模型和已加载的模型列表。"""
    mgr = _get_manager()
    return JSONResponse(content={
        "code": 0,
        "message": "success",
        "data": {
            "default_model": settings.DEFAULT_MODEL,
            "loaded_models": mgr.loaded_models,
        },
    })
