import torch
from torch.utils.data import Dataset


class MemDataset(Dataset):
    def __init__(self, filename, transform=None) -> None:
        super().__init__()
        data = torch.load(filename)
        self.transform = transform
        self.input = data["input"]
        self.mem = data["mem"]
        self.spike = data["spike"]

    def __len__(self):
        return self.input.size(0)

    def __getitem__(self, index):
        x = self.input[index]
        m = self.mem[index]
        s = self.spike[index]
        if self.transform:
            x = self.transform(x)
        return x, m, s
