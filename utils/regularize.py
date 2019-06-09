import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import grad
from collections import OrderedDict


def regularize(model, optim, loss_fn, reg, reg_coef):
    """Regularize model and loss function.

    Args:
        model (nn.Module): pytorch model
        loss_fns (list of loss_fn): List of loss function
        reg (torch.Tensor): regularization loss with be kept in this tensor.
        reg_coef (scalar): regularization coefficient.
    Returns:
        tuple: model, reg_loss_fns
    """

    def reg_hook(module, input, output):
        """To add regularization loss dynamically.

        Args:
            module (nn.Module): module to regularize
            input (tuple): input[0].shape = [batch_size, num_channels, h, w]
            output (torch.Tensor)
        """
        nonlocal reg
        # only add regularizer when input and output have the same shape
        if input[0].shape == output.shape:
            v = torch.randn_like(output)  # shape: [batch_size, num_channels, h, w]
            vjp = grad(output, input[0], grad_outputs=v, retain_graph=True)[
                0]  # shape: [batch_size, num_channels, h, w]
            vjv = v * vjp  # shape: [batch_size, num_channels, h, w]
            data_dims = list(range(1, len(vjv.shape)))
            regs = vjv.sum(dim=data_dims) - (v ** 2).sum(dim=data_dims)
            reg += F.relu(regs).mean()

    for m in model.modules():
        if 'Block' in m.__class__.__name__:
            m.register_forward_hook(reg_hook)

    def reg_random(model, input, output):
        nonlocal reg
        tmp = model._forward_hooks
        model._forward_hooks = OrderedDict()

        i = input[0]
        random_data = torch.rand_like(i).to(i)
        # print('reg before', reg)
        _ = model(random_data)
        # print('reg after', reg)

        model._forward_hooks = tmp

    model.register_forward_hook(reg_random)

    def reg_loss_fn(y_, y):
        nonlocal reg
        reg_loss = reg_coef * reg + loss_fn(y_, y)
        reg = 0
        return reg_loss

    return model, reg_loss_fn
