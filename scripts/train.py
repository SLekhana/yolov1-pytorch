import typer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from yolov1.engine.trainer import YOLOv1Module

app = typer.Typer()


@app.command()
def train(
    data_root: str = "data/VOC",
    epochs: int = 135,
    batch_size: int = 16,
    lr: float = 1e-3,
    num_workers: int = 4,
    precision: str = "16-mixed",
    ckpt_dir: str = "checkpoints",
    compile_model: bool = False,
    wandb: bool = False,
    wandb_project: str = "yolov1-pytorch",
):
    model = YOLOv1Module(
        data_root=data_root,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        compile_model=compile_model,
    )
    logger = WandbLogger(project=wandb_project) if wandb else CSVLogger("logs")
    trainer = pl.Trainer(
        max_epochs=epochs,
        precision=precision,
        callbacks=[
            ModelCheckpoint(dirpath=ckpt_dir, monitor="val_loss", save_top_k=3),
            LearningRateMonitor(),
        ],
        logger=logger,
        log_every_n_steps=10,
    )
    trainer.fit(model)


if __name__ == "__main__":
    app()
