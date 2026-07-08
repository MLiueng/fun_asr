"""Paraformer-zh 语音识别模型实现。"""

import logging
import os
import time
from typing import Any

import torch
from funasr import AutoModel

from .base import BaseASRModel, convert_audio_to_wav, remove_repeated_text

logger = logging.getLogger(__name__)


class ParaformerZHModel(BaseASRModel):
    """基于 FunASR paraformer-zh 的中文语音识别模型。

    自动携带 VAD (fsmn-vad) 和标点 (ct-punc) 子模型，
    支持任意长度音频输入。
    """

    def __init__(self, device: str = "cuda:0", model_id: str | None = None):
        from ..core.config import settings

        self._device = device
        self._model_id = model_id or settings.PARAFORMER_MODEL_ID
        self._model = None

    @property
    def name(self) -> str:
        return "paraformer-zh"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            logger.info(f"[{self.name}] Model already loaded, skipping.")
            return
        logger.info(f"[{self.name}] Loading model ({self._model_id}) on {self._device} ...")
        t0 = time.time()
        self._model = AutoModel(
            model=self._model_id,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            punc_model="ct-punc",
            device=self._device,
            trust_remote_code=True,
        )
        logger.info(f"[{self.name}] Model loaded in {time.time() - t0:.2f}s.")

    def unload(self) -> None:
        if self._model is not None:
            logger.info(f"[{self.name}] Unloading model ...")
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info(f"[{self.name}] Model unloaded, GPU cache cleared.")
        else:
            logger.warning(f"[{self.name}] Model not loaded, nothing to unload.")

    def transcribe(self, audio_path: str, **kwargs: Any) -> str:
        if self._model is None:
            raise ValueError(f"[{self.name}] Model not loaded. Call load() first.")

        # 音频预处理：非 wav 格式统一转换为 16kHz mono wav
        wav_path, is_temp = convert_audio_to_wav(audio_path, sample_rate=16000, mono=True)

        logger.info(f"[{self.name}] Transcribe: {wav_path}")
        t0 = time.time()

        try:
            hotword = kwargs.get("hotword", "")
            generate_kwargs = {
                "input": wav_path,
                "batch_size_s": 300,
            }
            if hotword:
                generate_kwargs["hotword"] = hotword

            result = self._model.generate(**generate_kwargs)
        finally:
            if is_temp:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        elapsed = time.time() - t0

        text = ""
        if result and len(result) > 0:
            raw = result[0].get("text", "")
            text = "".join(raw) if isinstance(raw, list) else raw

        # 后处理：去除重复文本
        text = remove_repeated_text(text.strip())
        logger.info(f"[{self.name}] Done in {elapsed:.2f}s, text: {text[:100]}")
        return text
