from .base import *
from torch import nn
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        self.norm2 = norm(dim)
        self.conv2 = conv3x3(dim, dim)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm3(out)
        return out


class ND_ODEfunc(nn.Module):
    '''
    Jacobian is negative definite
    '''

    def __init__(self, dim):
        super(ND_ODEfunc, self).__init__()
        #self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(dim, dim)
        #self.norm2 = norm(dim)
        # self.conv2 = conv3x3(dim, dim)
        #self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = x
        #out = self.norm1(x)
        #out = self.relu(out)
        out = self.conv1(out)
        #out = self.norm2(out)
        out = self.relu(out)
        out = -F.conv_transpose2d(out, self.conv1.weight, padding=1)
        #out = self.norm3(out)
        return out


class AS_ODEfunc(nn.Module):
    '''
    Jacobian is anti-symmetric
    input xp = (x, p)

    Here note that with
    $\dot Y = K_1^T \sigma(K_2 Z+b_1)$
    $\dot Z = -K_2^T \sigma(K_1 Y+b2)$
    we can also prove that Jacobian is antisymmetric
    '''

    def __init__(self, dim):
        super(AS_ODEfunc, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        dim = dim // 2

        #elf.norm1_x = norm(dim)
        self.conv1_x = conv3x3(dim, dim)
        #self.norm2_x = norm(dim)
        #elf.norm3_x = norm(dim)

        #elf.norm1_p = norm(dim)
        self.conv1_p = conv3x3(dim, dim)
        #self.norm2_p = norm(dim)
        #elf.norm3_p = norm(dim)

        self.nfe = 0

    def forward(self, t, xp):
        self.nfe += 1

        x, p = xp[:, :xp.shape[1]//2, :, :], xp[:, xp.shape[1]//2:, :, :]
        #x, p = self.norm1_x(p), self.norm1_p(x)
        #x, p = self.relu(x), self.relu(p)
        x, p = self.conv1_x(p), self.conv1_p(x)
        #x, p = self.norm2_x(x), self.norm2_p(p)
        x, p = self.relu(x), self.relu(p)

        x, p = -F.conv_transpose2d(x, self.conv1_x.weight, padding=1),\
               -F.conv_transpose2d(p, self.conv1_p.weight, padding=1)
        # , p = self.norm3_x(x), self.norm3_p(p)
        x, p = x, -p

        xp = torch.cat([x, p], dim=1)
        return xp


class ODEBlock(nn.Module):

    def __init__(self, odefunc, tol, method):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.rtol = tol
        self.atol = tol
        self.method = method

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(
            x[0] if isinstance(x, tuple) else x)
        out = odeint(self.odefunc, x, self.integration_time,
                     rtol=self.rtol, atol=self.atol, method=self.method)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class ODENet(Model):
    def __init__(self, downscale_method, backbone_block, classify_method, int_method, tol, C):

        downscale = {
            'conv': [
                nn.Conv2d(1, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
            ],
            'res': [
                nn.Conv2d(1, 4, 3, 1),
                ResBlock(4, 16, stride=2, downsample=conv1x1(4, 16, 2)),
                ResBlock(16, 64, stride=2, downsample=conv1x1(16, 64, 2)),
            ],
            'squeeze': [
                ODEBlock(ND_ODEfunc(1), tol, method),
                SqueezeLayer(2),
                CropLayer(1),
                ODEBlock(ND_ODEfunc(4), tol, method),
                SqueezeLayer(2),
                ODEBlock(ND_ODEfunc(16), tol, method),
                SqueezeLayer(2),
            ]
        }[downscale_method]

        feature = {
            'odenet': [ODEBlock(ODEfunc(64), tol, method)],
            'odenetnd': [ODEBlock(ND_ODEfunc(64), tol, method)],
            'odenetas': [ODEBlock(AS_ODEfunc(64), tol, method)],
            'resnet': [ResBlock(64, 64) for _ in range(6)],
            'null': [],
        }[backbone_block]

        fc = {
            'LR': [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)],
            'LDA': [nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), ODEBlock(AS_ODEfunc(64), tol, method), Flatten()],
        }[classify_method]

        self.layers = nn.Sequential(downscale + feature + fc)

    def forward(self, x):
        return self.layers(x)
