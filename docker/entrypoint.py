"""Fun-ASR 离线部署启动脚本。
同进程内拦截 FunASR/ModelScope 联网请求，确保猴子补丁生效。
"""
import os
import sys

# ── 1. 在任何 import 之前，强制设置所有离线环境变量 ──
offline_env = {
    "MODELSCOPE_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
for k, v in offline_env.items():
    os.environ[k] = v
    print(f"[entrypoint] Set {k}={v}")

# ── 2. 猴子补丁：让 AutoModel 强制 disable_update=True ──
try:
    from funasr import AutoModel

    _orig_init = AutoModel.__init__

    def _patched_init(self, *args, **kwargs):
        kwargs["disable_update"] = True
        print("[entrypoint] AutoModel.__init__: disable_update=True forced")
        _orig_init(self, *args, **kwargs)

    AutoModel.__init__ = _patched_init
    print("[entrypoint] AutoModel patched: disable_update=True forced")
except Exception as e:
    print(f"[entrypoint] Warning: AutoModel patch failed: {e}")

# ── 3. 猴子补丁：拦截 ModelScope snapshot_download ──
try:
    import modelscope.hub.snapshot_download as _sd_mod

    _orig_snapshot_download = _sd_mod.snapshot_download

    def _patched_snapshot_download(*args, **kwargs):
        # 离线模式下：如果本地有缓存就直接返回路径，否则抛错
        kwargs["local_files_only"] = True
        print(f"[entrypoint] snapshot_download: forced local_files_only=True")
        return _orig_snapshot_download(*args, **kwargs)

    _sd_mod.snapshot_download = _patched_snapshot_download
    print("[entrypoint] ModelScope snapshot_download patched: local_files_only=True forced")
except Exception as e:
    print(f"[entrypoint] Warning: ModelScope snapshot_download patch failed: {e}")

# ── 4. 同进程启动 app.main（关键！subprocess 会使补丁失效）───
print(f"[entrypoint] Starting app.main with args: {sys.argv[1:]}")

# 模拟命令行参数
sys.argv = ["app.main"] + sys.argv[1:]

from app.main import main

main()
