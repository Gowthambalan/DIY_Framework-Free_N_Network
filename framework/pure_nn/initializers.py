from abc import ABC
import random
from .activations import Activation, ReLU
import math


def supported_initializers():
    return [x.__name__ for x in Initializer.__subclasses__()]


class Initializer:
    def initialize(self, x):
        raise NotImplementedError


class Constant(Initializer, ABC):
    def __init__(self, c=0):
        self._c = c

    def initialize(self, x):
        if isinstance(x[0], int):
            w = len(x)
            temp = [None for _ in range(w)]
            for i in range(w):
                temp[i] = self._c
            return temp
        elif isinstance(x[0], list):
            w, h = len(x), len(x[0])
            temp = [[None for _ in range(h)] for _ in range(w)]
            for i in range(w):
                for j in range(h):
                    temp[i][j] = self._c
            return temp
        else:
            raise TypeError


class RandomUniform(Initializer, ABC):
    def initialize(self, x):
        w, h = len(x), len(x[0])
        temp = [[None for _ in range(h)] for _ in range(w)]
        for i in range(w):
            for j in range(h):
                temp[i][j] = random.uniform(0, 1)
        return temp


class XavierUniform(Initializer, ABC):
    def initialize(self, x):
        fan_in, fan_out = len(x), len(x[0])  # TODO: only works for Dense layer!
        std = math.sqrt(2 / (fan_in + fan_out))
        a = std * math.sqrt(3)

        temp = [[None for _ in range(fan_out)] for _ in range(fan_in)]
        for i in range(fan_in):
            for j in range(fan_out):
                temp[i][j] = random.uniform(-a, a)
        return temp


class HeNormal(Initializer, ABC):
    def __init__(self, non_linearity, mode="fan_in"):
        if not isinstance(non_linearity, Activation):
            raise Exception()
        self.non_linearity = non_linearity
        self.mode = mode

    def initialize(self, x):
        fan_in, fan_out = len(x), len(x[0])  # TODO: only works for Dense layer!
        fan = fan_in if self.mode == "fan_in" else fan_out
        if isinstance(self.non_linearity, ReLU):
            gain = math.sqrt(2)
        else:
            raise NotImplementedError
        std = gain / math.sqrt(fan)

        temp = [[None for _ in range(fan_out)] for _ in range(fan_in)]
        for i in range(fan_in):
            for j in range(fan_out):
                temp[i][j] = random.gauss(0, std)
        return temp
