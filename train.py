import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from loguru import logger

from dataset import MemDataset
from model import SDN
from utils import Metrics, MetricsCheckpoint, parameters_count

parser = argparse.ArgumentParser(description="SDN Training")
parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay")
parser.add_argument("--epochs", default=100, type=int, help="Training epochs")
parser.add_argument(
    "--num_workers", default=4, type=int, help="Number of workers to use for dataloader"
)
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
# Model
parser.add_argument("--n_layers", default=1, type=int, help="Number of layers")
parser.add_argument("--d_model", default=8, type=int, help="Model dimension")
parser.add_argument("--k", "--kernel_size", default=8, type=int, help="Kernel size")

# Dataset
parser.add_argument(
    "--training", type=str, required=True, help="Path to training dataset"
)
parser.add_argument("--test", type=str, required=True, help="Path to test dataset")

# General
items_group = parser.add_mutually_exclusive_group()
items_group.add_argument(
    "--resume", "-r", default=None, type=str, help="Path where checkpoint to resume"
)
items_group.add_argument(
    "--save",
    "-s",
    default="exp",
    type=str,
    help="Path where checkpoint to save and will be inactive given `resume`",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Model will be trained on device: {device}")

if args.resume is not None:
    args.save = args.resume
    
logger.add(os.path.join(args.save, "exp.log"))

# Data
logger.info("==> Preparing data.")


def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0 - val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val


transform = transforms.Lambda(lambda x: x.view(1, -1))

transform_train = transform_test = transform

trainset = MemDataset(
    filename=args.training,
    transform=transform_train,
)
trainset, valset = split_train_val(trainset, val_split=0.1)

testset = MemDataset(
    filename=args.test,
    transform=transform_test,
)

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

# Model
logger.info("==> Building model.")
model = SDN(
    d_model=args.d_model,
    kernel_size=args.k,
    n_layers=args.n_layers,
)

logger.info(model)
logger.info(f"Params: {parameters_count(model):,}")

model = model.to(device)
if device == "cuda":
    cudnn.benchmark = True

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)


best_metrics_checkpoint = MetricsCheckpoint(loss=float("+inf"), epoch=-1)
start_epoch = 0

last_checkpoint_filename = os.path.join(args.save, "last_checkpoint.pth")
best_checkpoint_filename = os.path.join(args.save, "best_checkpoint.pth")

if args.resume:
    # Load checkpoint.
    logger.info("==> Resuming from checkpoint..")
    assert os.path.exists(last_checkpoint_filename), "Error: no checkpoint found!"
    checkpoint = torch.load(last_checkpoint_filename)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    if os.path.exists(last_checkpoint_filename):
        best_metrics_checkpoint = torch.load(best_checkpoint_filename)["metrics"]


# Training
def train(trainloader):
    model.train()
    train_loss = Metrics("Loss")
    abs_error = []
    acc1 = Metrics("Acc@1", scale=100, format=".2f", suffix="%")
    for inputs, targets, spikes in trainloader:
        inputs, targets, spikes = (
            inputs.to(device),
            targets.to(device),
            spikes.to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        pred_s = (outputs.detach() + inputs.squeeze() >= 1).float()
        acc = (pred_s == spikes).float().mean()
        acc1.update(acc.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

        abs_error.append((outputs - targets).abs())
        train_loss.update(loss.item())
    abs_error = torch.cat(abs_error)
    std, mean = torch.std_mean(abs_error)
    return MetricsCheckpoint(
        loss=train_loss.avg,
        mae_max=abs_error.max().item(),
        mae_mean=mean.item(),
        mae_std=std.item(),
        acc1=acc1.avg,
    )


@torch.no_grad()
def eval(dataloader):
    model.eval()
    eval_loss = Metrics("Loss")
    abs_error = []
    acc1 = Metrics("Acc@1", scale=100, format=".2f", suffix="%")
    for inputs, targets, spikes in dataloader:
        inputs, targets, spikes = (
            inputs.to(device),
            targets.to(device),
            spikes.to(device),
        )
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        pred_s = (outputs.detach() + inputs.squeeze() >= 1).float()

        acc = (pred_s == spikes).float().mean()
        acc1.update(acc.item(), inputs.size(0))

        abs_error.append((outputs - targets).abs())
        eval_loss.update(loss.item())

    abs_error = torch.cat(abs_error)
    std, mean = torch.std_mean(abs_error)

    return MetricsCheckpoint(
        loss=eval_loss.avg,
        mae_max=abs_error.max().item(),
        mae_mean=mean.item(),
        mae_std=std.item(),
        acc1=acc1.avg,
    )


logger.info("==> Training")
for epoch in range(start_epoch, args.epochs):
    logger.info(f"==> Epoch {epoch}:")
    train_metrics_checkpoint = train(trainloader)
    logger.info(f"training:   {train_metrics_checkpoint}")
    val_metrics_checkpoint = eval(valloader)
    logger.info(f"validation: {val_metrics_checkpoint}")
    test_metrics_checkpoint = eval(testloader)
    logger.info(f"test:       {test_metrics_checkpoint}")

    scheduler.step()
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": val_metrics_checkpoint,
        "epoch": epoch,
    }
    torch.save(checkpoint, last_checkpoint_filename)
    if val_metrics_checkpoint < best_metrics_checkpoint:
        best_metrics_checkpoint = val_metrics_checkpoint
        checkpoint["test_metrics"] = test_metrics_checkpoint
        checkpoint["training_metrics"] = test_metrics_checkpoint
        torch.save(checkpoint, best_checkpoint_filename)

best_checkpoint = torch.load(best_checkpoint_filename)
logger.info("=================================")
logger.info("Best Performance:")
logger.info(f"Training:   {best_checkpoint['training_metrics']}")
logger.info(f"Validation: {best_checkpoint['metrics']}")
logger.info(f"Test:       {best_checkpoint['test_metrics']}")
logger.info("=================================")
logger.info("==> Finished.")
