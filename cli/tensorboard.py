from __future__ import annotations

import subprocess


def run_tensorboard() -> int:
    command = [
        "tensorboard",
        "--logdir",
        "./",
        "--samples_per_plugin",
        "scalars=0,images=0",
    ]
    return subprocess.run(command, check=False).returncode
