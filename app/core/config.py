"""全局配置。"""

import os


class Settings:
    """应用配置，支持环境变量覆盖。"""

    # 服务
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "9000"))
    DEVICE: str = os.getenv("DEVICE", "cuda:0")

    # 默认模型
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "paraformer-zh")

    # 日志
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "/app/logs")

    # 模型路径配置（支持本地路径，留空使用默认 ModelScope Repo ID）
    PARAFORMER_MODEL_ID: str = os.getenv("PARAFORMER_MODEL_ID", "paraformer-zh")


settings = Settings()
