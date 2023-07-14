from abc import ABC
import numpy as np


def supported_activations():
    return [x.__name__ for x in Activation.__subclasses__()]


class Activation:
    def __call__(self, x):
        return self.forward(x)

    @staticmethod
    def forward(x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Linear(Activation, ABC):
    @staticmethod
    def forward(x):
        return x

    def derivative(self, x):
        return np.ones_like(x)


class ReLU(Activation, ABC):
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


class Tanh(Activation, ABC):
    @staticmethod
    def forward(x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - self.forward(x) ** 2


class Sigmoid(Activation, ABC):
    # https://stackoverflow.com/a/23194336/12732481
    @staticmethod
    def forward(x):
        """Numerically stable sigmoid function."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.forward(x) * (1 - self.forward(x))
