import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl


@hydra.main(config_path="conf", config_name="train")
def train(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    callbacks = [hydra.utils.instantiate(callback) for callback in cfg.callbacks]

    trainer = pl.Trainer(callbacks=callbacks, **cfg.trainer)

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    train()
