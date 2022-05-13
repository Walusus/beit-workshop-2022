from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 256,
        num_workers: int = 8,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # pobieranie zbioru danych
        MNIST(self.data_dir, train=True, download=True)

    def setup(self, stage=None):
        # utworzenie obiektów Dataset dla zbioru treningowego i walidacyjnego 
        dataset = MNIST(self.data_dir, train=True, transform=self.transform)
        self.dataset_train, self.dataset_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        # utworzenie obiektu Dataloader dla danych treningowych
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        # utworzenie obiektu Dataloader dla danych walidacyjnych
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
