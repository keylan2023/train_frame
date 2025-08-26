from torch.utils.data import DataLoader

class DataModule:
    """Minimal DataModule interface."""
    def train_dataloader(self) -> DataLoader: raise NotImplementedError
    def val_dataloader(self) -> DataLoader: raise NotImplementedError
    def test_dataloader(self) -> DataLoader: raise NotImplementedError
