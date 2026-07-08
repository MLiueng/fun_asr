"""模型管理器 — 负责模型注册、生命周期和策略分发。"""

import logging
from typing import Dict, Type

from ..models.base import BaseASRModel
from ..models.paraformer import ParaformerZHModel
from .config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 模型注册表: 模型名 → 模型类
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type[BaseASRModel]] = {
    "paraformer-zh": ParaformerZHModel,
}


def register_model(name: str, cls: Type[BaseASRModel]) -> None:
    """注册新模型到全局注册表，方便后续扩展。"""
    if name in _MODEL_REGISTRY:
        logger.warning(f"Model '{name}' already registered, overwriting.")
    _MODEL_REGISTRY[name] = cls
    logger.info(f"Registered model: {name} -> {cls.__name__}")


class ModelManager:
    """管理所有已注册模型的实例和生命周期。

    用法:
        manager = ModelManager(device="cuda:0")
        manager.load("paraformer-zh")
        text = manager.transcribe("paraformer-zh", "/path/to/audio.wav")
        manager.unload("paraformer-zh")
    """

    def __init__(self, device: str | None = None):
        self.device = device or settings.DEVICE
        self._instances: Dict[str, BaseASRModel] = {}

    @property
    def available_models(self) -> list[str]:
        """所有已注册的模型名。"""
        return list(_MODEL_REGISTRY.keys())

    @property
    def loaded_models(self) -> list[str]:
        """当前已加载的模型名。"""
        return [name for name, inst in self._instances.items() if inst.is_loaded]

    def get_instance(self, name: str) -> BaseASRModel | None:
        """获取模型实例（可能未加载）。"""
        return self._instances.get(name)

    def _create_instance(self, name: str) -> BaseASRModel:
        """创建模型实例（不重复创建）。"""
        if name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{name}'. Available: {self.available_models}"
            )
        if name not in self._instances:
            cls = _MODEL_REGISTRY[name]
            self._instances[name] = cls(device=self.device)
            logger.info(f"Created instance for '{name}' on {self.device}")
        return self._instances[name]

    def load(self, name: str | None = None) -> BaseASRModel:
        """加载指定模型，默认加载 DEFAULT_MODEL。"""
        name = name or settings.DEFAULT_MODEL
        inst = self._create_instance(name)
        if not inst.is_loaded:
            inst.load()
        return inst

    def unload(self, name: str | None = None) -> None:
        """卸载指定模型，默认卸载 DEFAULT_MODEL。"""
        name = name or settings.DEFAULT_MODEL
        inst = self._instances.get(name)
        if inst is None:
            logger.warning(f"Model '{name}' not found in instances.")
            return
        inst.unload()

    def unload_all(self) -> None:
        """卸载所有已加载的模型。"""
        for name in list(self._instances.keys()):
            self.unload(name)

    def transcribe(self, name: str, audio_path: str, **kwargs) -> str:
        """按模型名分发识别调用。"""
        inst = self._instances.get(name)
        if inst is None or not inst.is_loaded:
            inst = self.load(name)
        return inst.transcribe(audio_path, **kwargs)
