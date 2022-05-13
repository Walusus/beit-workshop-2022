from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F

from project.modules.generator import Generator
from project.modules.discriminator import Discriminator


class GAN(LightningModule):
    def __init__(
        self,
        input_shape,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
    ):
        super().__init__()
        # zapisanie hiperparametrów w loggerze
        self.save_hyperparameters()

        self.latent_dim = latent_dim

        # inicjalizacja modułów sieci
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=input_shape)
        self.discriminator = Discriminator(img_shape=input_shape)

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # utworzenie szumu do próbkowania z przestrzeni ukrytej
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # trening generatora
        if optimizer_idx == 0:
            # generowanie obrazu na podstawie szumu
            gen_images = self(z)

            # utworzenie etykiety (obraz prawdziwy)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # wyliczenie straty generatora
            g_loss = F.binary_cross_entropy(self.discriminator(gen_images), valid)
            self.log("loss/generator", g_loss, prog_bar=True)
            return g_loss

        # trening dyskryminatora
        if optimizer_idx == 1:
            # utworzenie etykiety (obraz prawdziwy)
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # wyliczenie straty dyskryminatora dla obrazu prawdziwego
            real_loss = F.binary_cross_entropy(self.discriminator(imgs), valid)

            # utworzenie etykiety (obraz fałszywy)
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            # wygenerowanie obrazu na podstawie szumu
            gen_images = self(z).detach()

            # wyliczenie straty dyskryminatora dla obrazu prawdziwego
            fake_loss = F.binary_cross_entropy(self.discriminator(gen_images), fake)

            # wyliczenie ostatecznej straty dyskryminatora
            d_loss = (real_loss + fake_loss) / 2
            self.log("loss/discriminator", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # osobny ompitizer dla generatora i dyskryminatora
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        return [opt_g, opt_d], []
