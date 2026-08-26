from __future__ import annotations

import subprocess


def run_tensorboard() -> int:
    command = [
        "tensorboard",
        "--logdir",
        "./",
        "--samples_per_plugin",
        "scalars=20000,images=200,eq_curves=100,xy_curves=100",
    ]
    return subprocess.run(command, check=False).returncode
