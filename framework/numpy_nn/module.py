from .losses import Loss
from .layers import Layer, ParamLayer
from tabulate import tabulate
from copy import deepcopy as dc


class Module:
    def __init__(self):
        self._parameters = {}
        self._layers = []

    def __call__(self, x, eval=False):
        return self.forward(x, eval=eval)

    def forward(self, **kwargs):
        raise NotImplementedError

    @property
    def parameters(self):
        return self._parameters

    def __setattr__(self, key, value):
        if isinstance(value, Layer):
            layer = value
            if isinstance(layer, ParamLayer):
                self._parameters[key] = layer.vars
            self._layers.append(value)
        object.__setattr__(self, key, value)

    def backward(self, loss):
        assert isinstance(loss, Loss)
        delta = dict(delta=loss.delta)
        for layer in self._layers[::-1]:
            delta = layer.backward(**delta)

    def summary(self):
        print("\nModel Summary:")
        data = []
        name, output_shape, n_param = "Input", (None, self._layers[0].input_shape), 0
        data.append((name, output_shape, n_param))
        for i, layer in enumerate(self._layers):
            name, output_shape, n_param = layer.summary()
            name += f"[{i}]"
            data.append((name, output_shape, n_param))

        total_param = 0
        for x in data:
            *_, n_param = x
            total_param += n_param

        print(tabulate(data, headers=["Layer", "Output shape", "Param#"], tablefmt="grid"))
        print(f"total trainable parameters: {total_param}\n")

    def set_weights(self, params):
        copy_param = dc(params)
        self._parameters = dc(params)
        for i in range(len(self._layers)):
            if isinstance(self._layers[i], ParamLayer):
                k = list(copy_param.keys())[0]
                self._layers[i].vars = self._parameters[k]
                copy_param.pop(k)
