"""
To deal with CrossEntropyLoss and MMLDA_Loss in a unified way:
- the last layer of a model must be unnormalized, i.e. without log_softmax
- the loss must be CrossEntropyLoss, whose input is unnormalized logits.
- correpondingly, for `CallableModelWrapper` in cleverhans, output_layer='logits' 
"""
from .convnet import *
from .resnet import *
from .wideresnet import *
from .anode import *
from .nonexpansive import *
