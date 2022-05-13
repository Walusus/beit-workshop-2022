from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
import torchmetrics


class ValidationModel(LightningModule):
    def __init__(self, hidden_size=64, learning_rate=2e-4, input_dims=(1, 28, 28), num_classes=10):
        super().__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # parametry zbioru danych
        self.num_classes = num_classes
        self.dims = input_dims
        channels, width, height = self.dims

        # model PyTorch
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

        # precyzcja na zbiorze walidacyjnym
        self.val_accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)

        # logowanie straty treningowej
        self.log('loss/train', loss)
        # logowanie precyzji na zbiorze treningowym
        self.log('accuracy/train', torchmetrics.functional.accuracy(preds, y))

        return loss

    # metoda odpowiedzialna za walidacje
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        # aktualizacja precyzji na zbiorze walidacyjnym
        self.val_accuracy.update(preds, y)

        # logowanie straty na zbiorze walidacyjnym
        self.log("loss/val", loss)
        # logowanie precyzji na zbiorze walidacyjnym
        self.log("accuracy/val", self.val_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
