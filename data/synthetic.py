import torch
from torch.utils.data import Dataset, DataLoader, random_split

class _SyntheticCls(Dataset):
    def __init__(self, num_classes=3, samples_per_class=400, input_dim=32, seed=123):
        g = torch.Generator().manual_seed(seed)
        self.data = []
        self.targets = []
        for c in range(num_classes):
            mean = torch.randn(input_dim, generator=g) * 0.5 + (c * 2.0)
            cov = torch.eye(input_dim) * (0.8 + 0.2 * c)
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
            samples = dist.sample((samples_per_class,))
            self.data.append(samples)
            self.targets.append(torch.full((samples_per_class,), c))
        self.data = torch.vstack(self.data)
        self.targets = torch.cat(self.targets).long()

    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class ClassificationSynthetic:
    """Simple synthetic classification DataModule for quick tests."""
    def __init__(self, num_classes=3, samples_per_class=400, input_dim=32,
                 train_ratio=0.8, batch_size=64, num_workers=2, seed=123):
        self.ds = _SyntheticCls(num_classes, samples_per_class, input_dim, seed=seed)
        n_total = len(self.ds)
        n_train = int(n_total * train_ratio)
        n_val = n_total - n_train
        self.train_set, self.val_set = random_split(self.ds, [n_train, n_val],
                                                    generator=torch.Generator().manual_seed(seed))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()
