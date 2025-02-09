import torch


class piecewise_quadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
        return (x >= 0).to(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        x_abs = x.abs()
        mask = x_abs > 1
        grad_x = (grad_output * (-x_abs + 1.0)).masked_fill_(mask, 0)
        return grad_x, None


def piecewise_quadratic_surrogate():
    return piecewise_quadratic.apply
