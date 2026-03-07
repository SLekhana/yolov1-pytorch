from __future__ import annotations
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from yolov1.engine.trainer import YOLOv1Module


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lambda_coord = trial.suggest_float("lambda_coord", 3.0, 7.0)

    model = YOLOv1Module(
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        data_root="data/VOC",
        num_workers=4,
    )
    model.criterion.lambda_coord = lambda_coord

    trainer = pl.Trainer(
        max_epochs=5,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        callbacks=[EarlyStopping(monitor="val_loss", patience=2)],
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model)
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf")))
    return float(val_loss)


def run_search(n_trials: int = 20, output: str = "logs/hparam_results.json") -> None:
    import json
    from pathlib import Path

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    Path(output).parent.mkdir(exist_ok=True)
    Path(output).write_text(json.dumps({
        "best_params": study.best_params,
        "best_value": study.best_value,
        "trials": [{"params": t.params, "value": t.value} for t in study.trials],
    }, indent=2))

    print(f"\nBest params: {study.best_params}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    run_search()
