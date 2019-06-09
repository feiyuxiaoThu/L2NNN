import os
import torch
import yaml
import time
from easydict import EasyDict

import utils


class Config():
    """
    Customized config class that can save and load .yaml file.
    Can directly set and get attributes like EasyDict.

        Examples:
        >>> a = Config()
        >>> a = a.update(foo=1)
        >>> a.foo
        1
        >>> a.bar = 2
        >>> a.save('/tmp/config.yaml')
        >>> b = Config().load('/tmp/config.yaml')
        >>> b.bar
        2
        >>> b.get('nothing', False)
        False
    """

    def __init__(self):
        self._config = EasyDict()

    def __getattr__(self, attr):
        return getattr(self._config, attr)

    def __setattr__(self, attr, value):
        if not attr.startswith('_'):
            setattr(self._config, attr, value)
        else:
            self.__dict__[attr] = value

    def update(self, **kwargs):
        '''Update config from keyword arguments'''
        self._config.update(kwargs)
        return self

    def load(self, load_path):
        '''Load config from a file'''
        with open(load_path, 'r') as f:
            config = yaml.load(f)
        self.update(**config)
        return self

    def save(self, save_path):
        # convert easydict to dict
        config = eval(str(self._config))
        with open(save_path, 'w') as f:
            # to save with indent
            yaml.dump(config, f, default_flow_style=False)


# You can globally get config, but when needing to modify config,
# you need to pass config as a function argument for maintainability.
global config
config = Config()
