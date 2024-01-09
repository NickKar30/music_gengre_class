import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from music_genre_class.dataset import AudioDataModule
from music_genre_class.model import CNNModel


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    model = CNNModel(cfg)
    dm = AudioDataModule(
        dataset_path=cfg.data.dataset_path,
        url=cfg.data.url,
        val_size=cfg.data.val_size,
        batch_size=cfg.data.batch_size,
        genres=cfg.data.genres,
        num_waves=cfg.data.num_waves,
    )
    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.exp_name,
            tracking_uri="./logs/mlflow",
        ),
    ]

    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs, logger=loggers)

    trainer.fit(model, datamodule=dm)
    onnx_save_path = "data/onnx"
    os.makedirs(str(onnx_save_path), exist_ok=True)
    onnx_file_path = os.path.join(onnx_save_path, "model.onnx")
    model.to_onnx(onnx_file_path, export_params=True)


if __name__ == "__main__":
    main()
