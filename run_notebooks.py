"""Run project notebooks in the expected experiment order.

Default order:
1. 02_basic_strategies.ipynb
2. 03_mlp.ipynb
3. 04_lstm.ipynb

Use --skip-basic when 02_basic_strategies.ipynb has already finished and you
only want to continue with the MLP and LSTM notebooks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


NOTEBOOKS = [
    "02_basic_strategies.ipynb",
    "03_mlp.ipynb",
    "04_lstm.ipynb",
]


def run_notebook(notebook: Path, timeout: int, kernel: str | None) -> None:
    if not notebook.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook}")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    if kernel:
        cmd.append(f"--ExecutePreprocessor.kernel_name={kernel}")
    cmd.append(str(notebook))

    print(f"\n=== Running {notebook.name} ===", flush=True)
    subprocess.run(cmd, check=True)
    print(f"=== Finished {notebook.name} ===", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute 02_basic_strategies, then 03_mlp, then 04_lstm."
    )
    parser.add_argument(
        "--skip-basic",
        action="store_true",
        help="Start from 03_mlp.ipynb because 02_basic_strategies.ipynb already finished.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell timeout in seconds. Use -1 for no timeout.",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Optional Jupyter kernel name, for example python3.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    notebooks = NOTEBOOKS[1:] if args.skip_basic else NOTEBOOKS

    try:
        for name in notebooks:
            run_notebook(project_root / name, timeout=args.timeout, kernel=args.kernel)
    except subprocess.CalledProcessError as exc:
        print(f"\nStopped because a notebook failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode
    except Exception as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        return 1

    print("\nAll requested notebooks finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
