import os
import re
from typing import Optional


def _get_current_version(log_dir: str = "lightning_logs") -> int:
    """
    获取当前版本号。logs 目录默认为 ./lightning_logs
    """
    log_dir = os.path.join(os.getcwd(), log_dir)
    if not os.path.exists(log_dir):
        return 0
    listdir_info = [{"name": os.path.join(log_dir, d)} for d in os.listdir(log_dir)]

    existing_versions = []
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if os.path.isdir(d) and bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0

    return max(existing_versions)


def get_specified_version_name(version: Optional[int] = None, log_dir: str = "lightning_logs") -> str:
    """
    根据版本号获取版本名称。用于恢复训练。支持 version_001 这种形式

    若 version 为 None，则返回最新版本的名称
    """
    if version is None:
        version = _get_current_version(log_dir)

    log_dir_full = os.path.join(os.getcwd(), log_dir)
    listdir_info = [{"name": os.path.join(log_dir_full, d)} for d in os.listdir(log_dir_full)]

    pattern = re.compile(rf"^version_0*{version}(?:_|$)")
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if os.path.isdir(d) and pattern.match(bn):
            return bn
    raise ValueError(f"version {version} not found")


def format_next_version_name(suffix: str = "", version: int = None, log_dir: str = "lightning_logs") -> str:
    """
    格式化下一个版本的名称。格式为 version_{version}_{suffix}
    """
    next_version = _get_current_version(log_dir) + 1 if version is None else version
    next_version = str(next_version).zfill(3)
    return f"version_{next_version}_{suffix}" if suffix else f"version_{next_version}"
