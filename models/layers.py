import numpy as np
import torch
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        dim = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, dim)


def unsqueeze(input, upscale_factor=2):
    '''
    [:, C*r^2, H, W] -> [:, C, H*r, W*r]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor**2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(batch_size, out_channels, upscale_factor,
                                         upscale_factor, in_height, in_width)

    output = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


def squeeze(input, downscale_factor=2):
    '''
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    '''
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor
    )

    output = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width)


class SqueezeLayer(nn.Module):
    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._upsample(x, logpx)
        else:
            return self._downsample(x, logpx)

    def _downsample(self, x, logpx=None):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logpx is None:
            return squeeze_x
        else:
            return squeeze_x, logpx

    def _upsample(self, y, logpy=None):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logpy is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logpy


class CropLayer(nn.Module):
    '''
    [:, C, H, W] -> [:, C, H-2r, W-2r]
    '''

    def __init__(self, num_crop=1):
        super(CropLayer, self).__init__()
        self.num_crop = num_crop

    def forward(self, x):
        return x[:, :, self.num_crop:-self.num_crop, self.num_crop:-self.num_crop]


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def bn(dim):
    """GroupNorm with num_groups less than 32"""
    return nn.GroupNorm(min(32, dim), dim)


def get_means(C=1., p=784, L=10):
    """
    Generate MMD means.
    Input:  The constant C, the dimension of vectors p and the number of classes L. (L ≤ p + 1)
    """
    means = [torch.zeros(p) for _ in range(L)]
    means[0][0] = 1.
    for i in range(2, L+1):
        for j in range(1, i):
            means[i-1][j-1] = - (1 + torch.dot(means[i-1],
                                               means[j-1]) * (L-1)) / means[j-1][j-1] / (L-1)
        if i != L:
            means[i-1][i-1] = np.sqrt(1 - np.dot(means[i-1], means[i-1]))
    means = torch.cat([(np.sqrt(C) * mean).unsqueeze(0) for mean in means])

    return means


class MMLDA_Layer(nn.Module):
    """MMLDA_Layer

    Args:
        device (string): 'cpu' or 'cuda'
        C (float): L_2 norm of mean vectors
        sigma (float): variance of the MMLDA dist
        p (int): dimension of final features
        L (int): number of classes
    Examples:
        >>> import models, torch
        >>> m=models.layers.MMLDA_Layer(C=1, sigma=1, p=2, L=3)
        >>> x = torch.Tensor([[1,0], [-1, 0]]) 
        >>> assert m(x).shape == torch.Size([x.shape[0], m.L])
    """

    def __init__(self, device='cpu', C=1., sigma=1., p=784, L=10):
        super(MMLDA_Layer, self).__init__()
        self.C = C
        self.sigma = sigma
        self.p = p
        self.L = L
        self.means = get_means(C, p, L).to(device)
        self.out_features = L  # for cleverhans

    def forward(self, x):
        logits = -(x.unsqueeze(1).expand(-1, self.L, -1) - self.means).pow(2).sum(dim=2) / (self.sigma ** 2)
        return logits
