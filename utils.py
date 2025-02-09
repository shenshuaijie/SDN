from __future__ import annotations

import torch


def parameters_count(model: torch.nn.Module):
    return sum(p.nelement() for p in model.parameters())


class Metrics:
    def __init__(
        self, name: str, scale: float = 1.0, format=".4f", suffix: str = ""
    ) -> None:
        self.name = name
        self.scale = scale
        self.format = format
        self.suffix = suffix
        self.reset()

    def reset(self):
        self.val = 0
        self.num = 0

    def update(self, val: float, num: int = 1):
        self.val += val * num
        self.num += num

    def __lt__(self, other: Metrics):
        return self.avg < other.avg

    @property
    def avg(self):
        if self.num == 0:
            return 0.0
        return self.val / self.num

    def __str__(self):
        return f"{self.name}: {self.avg * self.scale:{self.format}}{self.suffix}"


class MetricsCheckpoint:
    def __init__(self, key: str = None, **kwargs) -> None:
        self.key = key
        self.data = kwargs

    def __lt__(self, other: MetricsCheckpoint):
        if self.key is not None and self.key in self.data:
            return self.data[self.key] < other.data[self.key]
        for key in self.data.keys():
            return self.data[key] < other.data[key]

    def __str__(self):
        return " | ".join(f"{k}: {v:.6f}" for k, v in self.data.items())


if __name__ == "__main__":
    model = torch.nn.Linear(1024, 1024)
    print(f"{parameters_count(model):,}")

    m1 = MetricsCheckpoint(
        accurcy=0.18,
        epoch=1,
    )
    m2 = MetricsCheckpoint(accurcy=0.32, epoch=2)
    print(m2)

    m = Metrics("accuracy", 100, suffix="%")
    m.update(0.9)
    print(m)
    m.update(0.8)
    print(m)
