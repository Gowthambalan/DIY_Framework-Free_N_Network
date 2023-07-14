from abc import ABC
import numpy as np


def supported_optimizers():
    return [x.__name__ for x in Optimizer.__subclasses__()]


class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def apply(self):
        raise NotImplementedError


class SGD(Optimizer, ABC):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr)

    def apply(self):
        for param in self.params.values():
            param["W"] -= self.lr * param["dW"]
            param["b"] -= self.lr * param["db"]


class Momentum(Optimizer, ABC):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"gW": np.zeros_like(layer["dW"])})
            layer.update({"gb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["gW"] = param["dW"] + self.mu * param["gW"]
            param["W"] -= self.lr * param["gW"]
            param["gb"] = param["db"] + self.mu * param["gb"]
            param["b"] -= self.lr * param["gb"]


class RMSProp(Optimizer, ABC):
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": np.zeros_like(layer["dW"])})
            layer.update({"sb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["sW"] = self.beta * param["sW"] + (1 - self.beta) * np.square(param["dW"])
            param["W"] -= self.lr * param["dW"] / (np.sqrt(param["sW"]) + self.eps)
            param["sb"] = self.beta * param["sb"] + (1 - self.beta) * np.square(param["db"])
            param["b"] -= self.lr * param["db"] / (np.sqrt(param["sb"]) + self.eps)


class AdaGrad(Optimizer, ABC):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super(AdaGrad, self).__init__(params, lr)
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": np.zeros_like(layer["dW"])})
            layer.update({"sb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["sW"] = np.square(param["dW"])
            param["W"] -= self.lr * param["dW"] / (np.sqrt(param["sW"]) + self.eps)
            param["sb"] = np.square(param["db"])
            param["b"] -= self.lr * param["db"] / (np.sqrt(param["sb"]) + self.eps)


class Adam(Optimizer, ABC):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"mW": np.zeros_like(layer["dW"])})
            layer.update({"vW": np.zeros_like(layer["dW"])})
            layer.update({"mb": np.zeros_like(layer["db"])})
            layer.update({"vb": np.zeros_like(layer["db"])})

    def apply(self):
        for param in self.params.values():
            param["mW"] = (1 - self.beta1) * param["dW"] + self.beta1 * param["mW"]
            param["vW"] = (1 - self.beta2) * np.square(param["dW"]) + self.beta2 * param["vW"]
            mw_hat = param["mW"] / (1 - self.beta1 ** self.k)
            vw_hat = param["vW"] / (1 - self.beta2 ** self.k)
            param["W"] -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)

            param["mb"] = (1 - self.beta1) * param["db"] + self.beta1 * param["mb"]
            param["vb"] = (1 - self.beta2) * np.square(param["db"]) + self.beta2 * param["vb"]
            mb_hat = param["mb"] / (1 - self.beta1 ** self.k)
            vb_hat = param["vb"] / (1 - self.beta2 ** self.k)
            param["b"] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)
        self.k += 1
