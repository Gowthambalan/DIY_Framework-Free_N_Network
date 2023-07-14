import math
from abc import ABC
from .activations import Activation, ReLU
import numpy as np


def supported_initializers():
    return [x.__name__ for x in Initializer.__subclasses__()]


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class Constant(Initializer, ABC):
    def __init__(self, c=0):
        self._c = c

    def initialize(self, shape):
        return self._c * np.ones(shape)


class RandomUniform(Initializer, ABC):
    def initialize(self, shape):
        return np.random.uniform(0, 1, shape)


class XavierUniform(Initializer, ABC):
    def initialize(self, shape):
        fan_in, fan_out = shape
        std = math.sqrt(2 / (fan_in + fan_out))
        a = std * math.sqrt(3)
        return np.random.uniform(-a, a, shape)


class HeNormal(Initializer, ABC):
    def __init__(self, non_linearity, mode="fan_in"):
        if not isinstance(non_linearity, Activation):
            raise Exception()
        self.non_linearity = non_linearity
        self.mode = mode

    def initialize(self, shape):
        fan_in, fan_out = shape
        fan = fan_in if self.mode == "fan_in" else fan_out
        if isinstance(self.non_linearity, ReLU):
            gain = math.sqrt(2)
        else:
            raise NotImplementedError
        std = gain / math.sqrt(fan)
        return np.random.normal(0, std, shape)
