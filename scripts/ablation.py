from __future__ import annotations
import subprocess
import json
from pathlib import Path


ABLATIONS = [
    {"name": "baseline",    "lr": "1e-3", "precision": "32"},
    {"name": "fp16",        "lr": "1e-3", "precision": "16-mixed"},
    {"name": "fp16_lr1e-4", "lr": "1e-4", "precision": "16-mixed"},
]


def main() -> None:
    results = {}
    for cfg in ABLATIONS:
        print(f"\n=== Ablation: {cfg['name']} ===")
        subprocess.run([
            "python", "scripts/train.py",
            "--epochs", "10",
            "--precision", cfg["precision"],
            "--lr", cfg["lr"],
            "--ckpt-dir", f"checkpoints/{cfg['name']}",
        ])
        results[cfg["name"]] = cfg
    Path("logs").mkdir(exist_ok=True)
    Path("logs/ablation_configs.json").write_text(json.dumps(results, indent=2))
    print("\nAblation configs saved to logs/ablation_configs.json")


if __name__ == "__main__":
    main()
