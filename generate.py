import torch
import os
import warnings
from torch import Tensor
import argparse


def heaviside(x: Tensor):
    """heaviside function

    Args:
        x (Tensor): u - vth

    Returns:
        Tensor: spike
    """
    return (x >= 0).int()


@torch.no_grad()
def hardreset(x: Tensor, tau: float = 0.2, v_th: float = 1.0):
    """perform lif evolution with hardreset mechanism

    Args:
        x (Tensor): input currents with (T, N)
        tau (float, optional): attenuation coefficient. Defaults to 0.2.
        v_th (float, optional): threshold when to spike. Defaults to 1.0.

    Returns:
        (Tensor, Tensor): spikes, attenuated membrane potential
    """
    x = x.cuda()
    if len(x.shape) == 1:
        x.view(-1, 1)
    y = []
    mem = []
    u = torch.zeros_like(x[0])
    for i in x:
        u = tau * u
        mem.append(u)
        u = u + i
        s = heaviside(u - v_th)
        y.append(s)
        u = u * (1 - s)
    y = torch.stack(y).int()
    mem = torch.stack(mem)
    return y, mem


@torch.no_grad()
def generate_dataset(
    root, name="training", number=5000, timestp=1024, m=0, std=1.0, tau=0.2
):
    """_summary_

    Args:
        root (path): the path to save dataset
        name (str, optional): the name of dataset. Defaults to "training".
        number (int, optional): sample number. Defaults to 5000.
        timestp (int, optional): steps. Defaults to 1024.
        m (int, optional): mean of input current. Defaults to 0.
        std (float, optional): std of input current. Defaults to 1.0.
        tau (float, optional): attenuation coefficient. Defaults to 0.2.
        v_th (float, optional): threshold when to spike. Defaults to 1.0.
    """
    filename = os.path.join(
        root, f"{name}-mem-T{timestp}-N({m},{std})-{number}-tau_{tau}.pt"
    )
    if os.path.exists(filename):
        warnings.warn(f"File `{filename}` exists, program terminated!")
        return

    os.makedirs(root, exist_ok=True)
    print("Sampling input currents ==>")
    dset_x = torch.randn(timestp, number) * std + m
    print("Generating label ==>")
    s, mem = hardreset(dset_x, tau=tau)
    print(f"spiking rate: {s.float().mean().item()}")
    data = {
        "input": dset_x.transpose(0, 1).cpu(),
        "mem": mem.transpose(0, 1).cpu(),
        "spike": s.transpose(0, 1).cpu(),
    }
    print("Dataset generated.")
    torch.save(data, filename)
    print(f"Dataset saved! File: {filename}")
    print(
        'File format: {"input": %s, "mem": %s, "spike": %s}'
        % (data["input"].size(), data["mem"].size(), data["spike"].size())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dateset generation script.")
    parser.add_argument("root", help="path to save dataset.")
    parser.add_argument("-n", "--name", default="training", type=str, help="name of dataset")
    parser.add_argument("-N", "--number", default=50000, type=int, help="size of dataset")
    parser.add_argument("-T", "--timestep", default=1024, type=int, help="number of step")
    parser.add_argument("-m", "--mean", default=0.0, type=float, help="mean of input current")
    parser.add_argument("-s", "--std", default=1.0, type=float, help="std of input current")
    parser.add_argument("-t", "--tau", default=0.2, type=float, help="attenuation coefficient")
    args = parser.parse_args()
    generate_dataset(args.root, args.name, args.number, args.timestep, args.mean, args.std, args.tau)
