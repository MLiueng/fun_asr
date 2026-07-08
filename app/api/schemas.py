"""API 请求/响应数据模型。"""

from typing import Any, Optional

from pydantic import BaseModel, Field


# ------ 通用响应 ------

class ApiResponse(BaseModel):
    """统一响应格式。"""
    code: int = Field(default=0, description="状态码，0 表示成功")
    message: str = Field(default="success", description="状态信息")
    data: Optional[Any] = Field(default=None, description="响应数据")


# ------ 语音识别 ------

class ASRUrlRequest(BaseModel):
    """URL 语音识别请求。"""
    url: str = Field(..., description="语音文件的 URL 地址")
