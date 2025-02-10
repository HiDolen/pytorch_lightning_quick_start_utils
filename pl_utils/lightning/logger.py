import os
import re
from typing import Optional


def get_current_version() -> int:
    """
    获取当前版本号。logs 目录为 ./lightning_logs
    """
    save_dir = os.path.join(os.getcwd(), "lightning_logs")
    if not os.path.exists(save_dir):
        return 0
    listdir_info = [{"name": os.path.join(save_dir, d)} for d in os.listdir(save_dir)]

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


def get_next_version() -> int:
    """
    获取下一个版本号
    """
    return get_current_version() + 1


def get_specified_version_name(version: Optional[int] = None) -> str:
    """
    根据版本号获取版本名称。用于恢复训练。支持 version_001 这种形式

    若 version 为 None，则返回最新版本的名称
    """
    if version is None:
        version = get_current_version()

    save_dir = os.path.join(os.getcwd(), "lightning_logs")
    listdir_info = [{"name": os.path.join(save_dir, d)} for d in os.listdir(save_dir)]

    pattern = re.compile(rf"^version_0*{version}")
    for listing in listdir_info:
        d = listing["name"]
        bn = os.path.basename(d)
        if os.path.isdir(d) and pattern.match(bn):
            return bn
    raise ValueError(f"version {version} not found")


def format_next_version_name(suffix: str = "", version: int = None) -> str:
    """
    格式化下一个版本的名称。格式为 version_{version}_{suffix}
    """
    next_version = get_next_version() if version is None else version
    next_version = str(next_version).zfill(3)
    return f"version_{next_version}_{suffix}" if suffix else f"version_{next_version}"
