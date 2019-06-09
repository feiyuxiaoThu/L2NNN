import os

import tensorflow as tf
import torch
from cleverhans.attacks import *
from cleverhans.model import CallableModelWrapper
from cleverhans.utils import AccuracyReport
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


class CleverhansAttack():
    '''
    Accepts a pytorch model and config, return an Attack object.

    Usage:
        attack = CleverhansAttack(model, config)
        adv_x = attack.perturb(x)
    '''

    def __init__(self, model, config):

        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
        self.sess = tf.Session(config=conf)

        im_shape = (None, 1, config.im_size, config.im_size)
        self.x_op = tf.placeholder(tf.float32, shape=im_shape)

        tf_model = convert_pytorch_model_to_tf(model)
        cleverhans_model = CallableModelWrapper(
            tf_model, output_layer='logits')
        attack = {
            'FGSM': FastGradientMethod,
            'PGD': ProjectedGradientDescent,
        }[config.attack]
        self.attack_op = attack(cleverhans_model, sess=self.sess)
        self.attack_params = config.attack_params

    def perturb_np(self, x):
        '''Returns a numpy array'''
        return self.attack_op.generate_np(x, **self.attack_params)

    def perturb(self, x):
        '''Returns a pytorch tensor'''
        return torch.Tensor(self.perturb_np(x))
