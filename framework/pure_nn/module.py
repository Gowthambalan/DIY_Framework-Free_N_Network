from .layers import Layer
from .losses import Loss


class Module:
    def __init__(self):
        self._parameters = {}
        self._layers = []

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    @property
    def parameters(self):
        return self._parameters

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            layer = value
            self._parameters[key] = layer.vars
            self._layers.append(value)
        object.__setattr__(self, key, value)

    def backward(self, loss):
        assert isinstance(loss, Loss)
        delta = loss.delta
        for layer in self._layers[::-1]:
            delta = layer.backward(delta)
