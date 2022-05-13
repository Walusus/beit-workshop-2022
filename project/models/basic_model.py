from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F


class BasicModel(LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, input_dims=(1, 28, 28), num_classes=10):
        super().__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Parametry zbioru danych
        self.num_classes = num_classes
        self.dims = input_dims
        channels, width, height = self.dims

        # Model PyTorch
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
