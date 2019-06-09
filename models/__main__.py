from torchsummary import summary

from .convnet import *
from .resnet import *
from .wideresnet import *
from .anode import *
from .nonexpansive import *
from config import config

if __name__ == '__main__':
    config = config.load('configs/mnist.yaml')
    config.time_dependent = False
    input_size = (config.in_channels, config.im_size, config.im_size)

    nets = [
        # ConvNet,
        # WideConvNet,
        # resnet20,
        # WideResNet,
        # ode_resnet20,
        L2NonExpaConvNet,
        L2NonExpa_resnet20,
    ]
    for net in nets:
        print(net)
        model = net(config).cuda()
        summary(model, input_size=input_size)
