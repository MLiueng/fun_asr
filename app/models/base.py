"""ASR 模型抽象基类 — 所有识别模型必须实现此接口。"""

import logging
import os
import re
import subprocess
import tempfile
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


def convert_audio_to_wav(
    audio_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[str, bool]:
    """将音频文件转换为指定采样率的 WAV 格式。

    使用 ffmpeg 进行转码，确保音频格式统一、采样率正确，
    这是提升 ASR 识别效果的关键预处理步骤。

    Args:
        audio_path: 原始音频文件路径
        sample_rate: 目标采样率，默认 16000Hz
        mono: 是否转为单声道，默认 True

    Returns:
        (转换后的文件路径, 是否为临时文件需要清理)
    """
    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".wav":
        # 即使是 wav，也检查是否需要重采样/转单声道
        return audio_path, False

    logger.info(f"[Audio] Converting {ext} to wav ({sample_rate}Hz {'mono' if mono else 'stereo'}) ...")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()

    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
    ]
    if mono:
        cmd.extend(["-ac", "1"])
    cmd.append(tmp_wav.name)

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return tmp_wav.name, True
    except Exception as e:
        logger.warning(f"[Audio] ffmpeg conversion failed: {e}, using original file")
        try:
            os.unlink(tmp_wav.name)
        except OSError:
            pass
        return audio_path, False


def remove_repeated_text(text: str, max_repeat: int = 3) -> str:
    """去除识别结果中的重复文本。

    ASR 模型有时会产生重复的短语（如"去去去那个"），此函数进行清理：
    - 连续重复的短词（1-4 字）保留最多 max_repeat 次
    - 去除完全重复的连续片段

    Args:
        text: 原始识别文本
        max_repeat: 允许的最大连续重复次数

    Returns:
        清理后的文本
    """
    if not text:
        return text

    # 去除连续重复的短词（1-4字符），如 "去去去那个" -> "去那个"
    pattern = rf"([\u4e00-\u9fff\w]{{1,4}})(\1){{{max_repeat},}}"
    text = re.sub(pattern, r"\1" * max_repeat, text)

    # 去除连续重复2次以上的短语（2-6字符）
    pattern2 = rf"([\u4e00-\u9fff\w]{{2,6}})(\1){{2,}}"
    text = re.sub(pattern2, r"\1\1", text)

    return text


class BaseASRModel(ABC):
    """语音识别模型抽象基类。

    每种模型实现需提供:
      - name:       模型唯一标识 (用于 API 路由分发)
      - load():     加载模型到 GPU/CPU
      - unload():   释放模型资源
      - is_loaded:  模型是否已加载
      - transcribe(): 对音频文件执行识别
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """模型唯一标识，例如 'paraformer-zh'。"""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """模型是否已加载到内存。"""
        ...

    @abstractmethod
    def load(self) -> None:
        """加载模型权重到设备。"""
        ...

    @abstractmethod
    def unload(self) -> None:
        """卸载模型并释放 GPU 显存。"""
        ...

    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        """对音频文件执行语音识别。

        Args:
            audio_path: 音频文件路径
            **kwargs:   模型特有参数 (如 language, hotwords 等)

        Returns:
            识别文本
        """
        ...
