from __future__ import annotations

import subprocess


def run_tensorboard() -> int:
    command = [
        "tensorboard",
        "--logdir",
        "./",
        "--samples_per_plugin",
        "scalars=20000,images=200,eq_curves=500,xy_curves=500",
    ]
    return subprocess.run(command, check=False).returncode
