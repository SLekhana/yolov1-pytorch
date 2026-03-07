from __future__ import annotations
import subprocess
import sys


def generate() -> None:
    subprocess.run([
        sys.executable, "-m", "pdoc",
        "--html",
        "--output-dir", "docs",
        "--force",
        "yolov1",
    ], check=True)
    print("Docs generated in docs/")


if __name__ == "__main__":
    generate()
