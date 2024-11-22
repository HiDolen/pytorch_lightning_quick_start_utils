import os


def get_next_version() -> int:
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

    return max(existing_versions) + 1

def format_next_version_name(suffix: str = "", version: int = None) -> str:
    next_version = get_next_version() if version is None else version
    next_version = str(next_version).zfill(3)
    return f"version_{next_version}_{suffix}" if suffix else f"version_{next_version}"

