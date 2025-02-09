"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.nn import DropoutNd
from src.models.spike.neuron import registry
from src.models.spike.surrogate import piecewise_quadratic_surrogate
from src.models.sequence.kernels.ssm import SSMKernelDiag


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, channels=1, lr=None):
        super().__init__()
        # Generate dt
        lr = min(lr, 0.001)
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        C = torch.randn(channels, H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (C H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("chn, hnl -> chl", C, torch.exp(K)).real

        return K, None

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class SS4D(nn.Module):
    def __init__(
        self,
        d_model,
        neuron="sdn",
        learnable_vth=True,
        shared_vth=False,
        d_state=64,
        dropout=0.0,
        transposed=True,
        bidirectional=False,
        channels=1,
        trainable_B=False,
        **kernel_args,
    ):
        super().__init__()

        self.h = d_model
        self.n = d_state
        # self.d_output = self.h
        self.transposed = transposed
        self.learnable_vth = learnable_vth
        self.bidirectional = bidirectional
        self.D = nn.Parameter(torch.randn(channels, self.h))

        if learnable_vth:
            if shared_vth:
                self.ln_vth = nn.Parameter(torch.zeros(1))
            else:
                self.ln_vth = nn.Parameter(torch.zeros(d_model, 1))

        if bidirectional:
            channels *= 2
        # SSM Kernel
        if trainable_B:
            self.kernel = SSMKernelDiag(d_model=self.h, d_state=self.n, channels=channels, init="diag-lin", **kernel_args)
        else:
            self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)

        self.neuron = registry[neuron](piecewise_quadratic_surrogate())

        # # Pointwise
        # self.activation = nn.GELU() no use

        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """Input and output shape (B, H, L)"""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k, _ = self.kernel(L=L)  # (H L)
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)

        y = torch.einsum("bhl,chl->bchl", u_f, k_f)

        y = torch.fft.irfft(y, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + torch.einsum("bhl,ch->bchl", u, self.D)

        y = rearrange(y, "b c h l -> b (c h) l")

        if self.learnable_vth:
            y = y / torch.exp(self.ln_vth)

        y = self.dropout(self.neuron(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return (
            y,
            None,
        )  # Return a dummy state to satisfy this repo's interface, but this can be modified


class SpikingSSM(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        norm="layer",
        layer=None,  # layer config
        **kwargs,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.norm = norm

        # for dataset adaptability
        self.d_model = self.d_output = d_model

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(SS4D(d_model, dropout=dropout, transposed=True, **layer))
            if norm == "batch":
                self.norms.append(nn.BatchNorm1d(d_model))
            elif norm == "layer":
                self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))

    def forward(self, x, **kwargs):
        """
        Input x is shape (B, L, d_input)
        """
        # x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            # Prenorm
            if self.prenorm:
                z = norm(z) if self.norm == "batch" else norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                z = norm(z) if self.norm == "batch" else norm(z.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        return x, None
