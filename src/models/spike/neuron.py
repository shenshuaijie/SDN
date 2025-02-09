from torch import nn
import torch
import os


class SDNNeuron(nn.Module):
    ckt_path = "mem_pred_d8k8_c3(relu(c2(c1(x))+c1(x)))_125.pkl"

    def __init__(self, surrogate_function):
        super().__init__()
        self.surrogate_function = surrogate_function
        ckt_path = os.path.join(os.path.dirname(__file__), self.ckt_path)
        self.model = torch.jit.load(ckt_path).eval()

    def forward(self, x):
        mem = self.pred(x)
        s = self.surrogate_function(mem + x - 1.0)
        return s

    @torch.no_grad()
    def pred(self, x):
        shape = x.shape
        L = x.size(-1)
        return self.model(x.detach().view(-1, 1, L)).view(shape)


class BPTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            u = u * self.tau + x[..., i]
            s = self.surrogate_function(u - self.vth)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)


class SLTTNueron(nn.Module):
    def __init__(self, surrogate_function, tau=0.125, vth=1.0, v_r=0):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.tau = tau
        self.vth = vth
        self.v_r = v_r

    def forward(self, x):
        u = torch.zeros_like(x[..., 0])
        out = []
        for i in range(x.size(-1)):
            u = u.detach() * self.tau + x[..., i]
            s = self.surrogate_function(u - self.vth)
            out.append(s)
            u = (1 - s.detach()) * u + s.detach() * self.v_r
        return torch.stack(out, -1)


registry = {
    "sdn": SDNNeuron,
    "bptt": BPTTNueron,
    "sltt": SLTTNueron,
}
