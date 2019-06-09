import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from utils import prod
from config import config


class CRelu(nn.Module):
    """
    Implement the Concatenated ReLU in https://arxiv.org/pdf/1603.05201.pdf
    """

    def __init__(self):
        super(CRelu, self).__init__()
        self.expansion = 2

    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)  # relu along channel


class NormPooling2d(nn.Module):
    def __init__(self, *k, **kw):
        super(NormPooling2d, self).__init__()
        self.avg_pool_kernel = nn.AvgPool2d(*k, **kw)

    def forward(self, x):
        ks = self.avg_pool_kernel.kernel_size
        coef = sqrt(prod(ks))
        # after average pooling the square we sqrt then mult kernel size.
        return coef * torch.sqrt(self.avg_pool_kernel(x * x))


class AdaptiveNormPool2d(nn.Module):
    def __init__(self, *k, **kw):
        super(AdaptiveNormPool2d, self).__init__()

    def forward(self, x):
        ks = list(x.shape)[2:]
        coef = sqrt(prod(ks))
        if not hasattr(self, 'pool'):
            self.pool = nn.AvgPool2d(ks)
        return coef * torch.sqrt(self.pool(x * x))


class NE_Conv2d(nn.Conv2d):
    '''Nonexpansive conv2d'''

    def __init__(self, *k, **kw):
        super(NE_Conv2d, self).__init__(*k, **kw)

    def forward(self, x):
        if config.div_before_conv:
            x = x / sqrt(prod(self.kernel_size) / prod(self.stride))
        return super(NE_Conv2d, self).forward(x)

    def L2Norm(self):
        if config.div_before_conv:
            return norm(self.weight)
        else:
            return norm(self.weight) * prod(self.kernel_size) / prod(self.stride)


class L2NonExpaNet(nn.Module):
    '''Base class of L2NNN'''

    def __init__(self):
        super(L2NonExpaNet, self).__init__()

    def forward(self, x):
        raise Exception('Not Implemented!')

    def L2Norm(self):
        '''Returns upper bound of L2Norm's square.'''
        raise Exception('Not Implemented!')

    def L2NNN_loss(self, pred, label):
        loss = sum([
            self.loss_a(pred, label) * self.a,
            self.loss_b(pred, label) * self.b,
            self.loss_c(pred, label) * self.c,
            self.loss_w() * self.w,
        ])
        return loss

    def normalize(self):
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, 'weight'):
                    m.weight /= norm(m.weight).sqrt()

    def c_gap(self, x, y):
        """Computes confidence gaps given input and label.

        Args:
            x (Tensor): input shape of [b, i, h, w]
            y (Tensor): label shape of [b]

        Returns:
            gap (torch.FloatTensor): shape [b]
        """
        pred = self.forward(x)
        v, i = pred.topk(2)
        # if prediction is wrong, gap is 0
        gap = (v[:, 0] - v[:, 1]) * (i[:, 0] == y).float() / self.L2Norm().sqrt()
        return gap

    def loss_a(self, pred, label):
        return F.cross_entropy(pred * self.u, label)

    def loss_b(self, pred, label):
        return F.cross_entropy(pred * self.v, label)

    def loss_c(self, pred, label):
        logits = F.softmax(pred * self.z)
        logit = logits.gather(1, label.view(-1, 1))
        return (1 - logit + 1e-10).log().mean() / self.z

    def loss_w(self):
        loss = 0
        for m in self.modules():
            if hasattr(m, 'weight'):
                loss += weight_reg_loss(m.weight)
        return loss


class L2NonExpaResBlock(nn.Module):
    '''L2 Nonexpansive version of PreAct ResBlock'''
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(L2NonExpaResBlock, self).__init__()
        self.act = CRelu()
        self.conv1 = NE_Conv2d(inplanes*self.act.expansion, planes, kernel_size=3, stride=stride, padding=1)
        self.conv2 = NE_Conv2d(planes*self.act.expansion, planes, kernel_size=3, padding=1)
        self.downsample = downsample
        self.stride = stride
        self.t = nn.Parameter(torch.tensor(1/2).sqrt(), requires_grad=True)

    def forward(self, x):
        residual = x

        out = self.act(x)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)

        out = out * self.t + residual * (1 - self.t ** 2).sqrt()
        return out

    def L2Norm(self):
        norm_o = self.conv1.L2Norm() * self.conv2.L2Norm()
        if self.downsample is not None:
            m = list(self.downsample.children())[0]
            norm_r = m.L2Norm()
        else:
            norm_r = 1
        norm_total = norm_o * self.t**2 + norm_r * (1-self.t**2)
        return norm_total


class L2NonExpaResNet(L2NonExpaNet):
    '''L2 Nonexpansive version of PreAct ResNet'''

    def __init__(self, block, layers, config):
        super(L2NonExpaResNet, self).__init__()
        self.inplanes = 16
        self.act = CRelu()
        self.conv1 = NE_Conv2d(config.in_channels, 16,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.pool = AdaptiveNormPool2d()
        self.fc = nn.Linear(64*block.expansion*self.act.expansion, config.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                NE_Conv2d(self.inplanes*self.act.expansion, planes*block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.act(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def L2Norm(self):
        l2_norm = self.conv1.L2Norm()
        for i in range(1, 4):
            for m in getattr(self, f'layer{i}').children():
                l2_norm *= m.L2Norm()
        l2_norm *= norm(self.fc.weight)
        return l2_norm


def L2NonExpa_resnet20(config):
    model = L2NonExpaResNet(L2NonExpaResBlock, [3, 3, 3], config)
    return model


class L2NonExpaConvNet(L2NonExpaNet):
    '''L2 Nonexpansive version of WideConvNet.'''

    def __init__(self, config):
        super(L2NonExpaConvNet, self).__init__()
        self.act = CRelu()
        self.pool = NormPooling2d((2, 2), stride=(2, 2), padding=0)
        self.conv1 = NE_Conv2d(config.in_channels, 32, 5, stride=1, padding=2, bias=True)
        self.conv2 = NE_Conv2d(32*self.act.expansion, 64, 5, stride=1, padding=2, bias=True)
        self.num_dense = (config.im_size // 4) ** 2 * 64 * self.act.expansion
        self.fc1 = nn.Linear(self.num_dense, 1024, bias=True)
        self.fc2 = nn.Linear(1024*self.act.expansion, config.num_classes)

        self.u = nn.Parameter(torch.rand(config.num_classes), requires_grad=True)
        self.v = nn.Parameter(torch.tensor(config.v), requires_grad=config.v_use_grad)
        self.z = torch.tensor(config.z, requires_grad=False)
        self.a, self.b, self.c, self.w = config.a, config.b, config.c, config.w
        # if self.multi, we treat last fc layer as outputing num_classes scalars.
        # else, we treat last fc layer as outputting a vector of size [b, num_classes].
        # self.multi = config.multi

    def forward(self, x):
        x = self.pool(self.act(self.conv1(x)))
        x = self.pool(self.act(self.conv2(x)))
        x = x.view(-1, self.num_dense)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

    def L2Norm(self):
        '''Computes L2Norm upper bound of the whole model'''
        l2_norm = prod([
            self.conv1.L2Norm(),
            self.conv2.L2Norm(),
            norm(self.fc1.weight),
            norm(self.fc2.weight),
        ])
        return l2_norm


def weight_reg_loss(w):
    '''Weight regularization in L2NNN paper.'''
    def reg(m):
        return F.relu(m.abs().sum(1) - 1).sum()
    w_ = w.reshape(w.shape[0], -1)
    m = w_.mm(w_.t())
    m_ = w_.t().mm(w_)
    return min(reg(m), reg(m_))


def norm(w):
    '''
    Upper bound of a matrix's L2 norm.
    # this bound is precise when input is scalar,
    # but is O(\sqrt(mn)) poor, m, n is the dim of input and output
    '''
    w_ = w.reshape(w.shape[0], -1)
    m = w_.mm(w_.t())
    m_ = w_.t().mm(w_)
    return min(r(m), r(m_))


def r(m):
    return m.abs().sum(1).max()
