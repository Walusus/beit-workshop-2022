import torch
import torchvision
from pytorch_lightnig.callbacks import Callback


class LogGeneratedImagesCallback(Callback):
  def __init__(self, n_images = 8):
    self.n_images = n_images

  def on_fit_start(self, trainer, pl_module):
    latent_dim = pl_module.latent_dim
    weight = pl_module.generator.model[0].weight

    self.z = torch.randn(self.n_images, latent_dim).type_as(weight)
    
  def on_train_epoch_end(self, trainer, pl_module) -> None:
    images = pl_module(self.z)
    grid = torchvision.utils.make_grid(images)
    trainer.logger.experiment.add_image("generated_images", grid, trainer.current_epoch)
