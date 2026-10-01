"""端到端演示：训练 + 多配置基准 + 旗舰 DiffuseFuse + torch 对拍，落盘 benchmark.json。

直接运行：``python diffusionforge/examples/run_demo.py``
（仓库根目录下，venv 解释器；会从仓库根把 diffusionforge 包加入 sys.path）
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from diffusionforge.cli import run_benchmark  # noqa: E402


def main():
    out = os.path.join(ROOT, "benchmark.json")
    run_benchmark(datasets=["gaussians25", "moons", "checkerboard"], out_path=out)


if __name__ == "__main__":
    main()
