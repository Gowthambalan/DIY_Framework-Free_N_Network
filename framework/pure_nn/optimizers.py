from abc import ABC
from .utils import *


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
            grad_step_w = rescale(param["dW"], -self.lr)
            param["W"] = mat_add(param["W"], grad_step_w)
            grad_step_b = rescale(param["db"], -self.lr)
            param["b"] = mat_add(param["b"], grad_step_b)


class Momentum(Optimizer, ABC):
    def __init__(self, params, lr, mu):
        super(Momentum, self).__init__(params, lr)
        self.mu = mu
        for layer in list(self.params.values()):
            layer.update({"gW": rescale(layer["dW"], 0)})
            layer.update({"gb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():
            param["gW"] = mat_add(param["dW"], rescale(param["gW"], self.mu))
            grad_step_w = rescale(param["gW"], -self.lr)
            param["W"] = mat_add(param["W"], grad_step_w)
            param["gb"] = mat_add(param["db"], rescale(param["gb"], self.mu))
            grad_step_b = rescale(param["gb"], -self.lr)
            param["b"] = mat_add(param["b"], grad_step_b)


class RMSProp(Optimizer, ABC):
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-8):
        super(RMSProp, self).__init__(params, lr)
        self.beta = beta
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": rescale(layer["dW"], 0)})
            layer.update({"sb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():

            grad_square_w = element_wise_mul(param["dW"], param["dW"])
            grad_square_w = rescale(grad_square_w, 1 - self.beta)
            param["sW"] = mat_add(rescale(param["sW"], self.beta), grad_square_w)
            grad_step_w = element_wise_mul(param["dW"], element_wise_rev(add_scalar(mat_sqrt(param["sW"]), self.eps)))
            param["W"] = mat_add(param["W"], rescale(grad_step_w, -self.lr))

            grad_square_b = element_wise_mul(param["db"], param["db"])
            grad_square_b = rescale(grad_square_b, 1 - self.beta)
            param["sb"] = mat_add(rescale(param["sb"], self.beta), grad_square_b)
            grad_step_b = element_wise_mul(param["db"], element_wise_rev(add_scalar(mat_sqrt(param["sb"]), self.eps)))
            param["b"] = mat_add(param["b"], rescale(grad_step_b, -self.lr))


class AdaGrad(Optimizer, ABC):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super(AdaGrad, self).__init__(params, lr)
        self.eps = eps
        for layer in list(self.params.values()):
            layer.update({"sW": rescale(layer["dW"], 0)})
            layer.update({"sb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():

            grad_square_w = element_wise_mul(param["dW"], param["dW"])
            param["sW"] = mat_add(param["sW"], grad_square_w)
            grad_step_w = element_wise_mul(param["dW"], element_wise_rev(add_scalar(mat_sqrt(param["sW"]), self.eps)))
            param["W"] = mat_add(param["W"], rescale(grad_step_w, -self.lr))

            grad_square_b = element_wise_mul(param["db"], param["db"])
            param["sb"] = mat_add(param["sb"], grad_square_b)
            grad_step_b = element_wise_mul(param["db"], element_wise_rev(add_scalar(mat_sqrt(param["sb"]), self.eps)))
            param["b"] = mat_add(param["b"], rescale(grad_step_b, -self.lr))


class Adam(Optimizer, ABC):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.k = 1
        for layer in list(self.params.values()):
            layer.update({"mW": rescale(layer["dW"], 0)})
            layer.update({"vW": rescale(layer["dW"], 0)})
            layer.update({"mb": rescale(layer["db"], 0)})
            layer.update({"vb": rescale(layer["db"], 0)})

    def apply(self):
        for param in self.params.values():
            param["mW"] = mat_add(rescale(param["dW"], 1 - self.beta1), rescale(param["mW"], self.beta1))
            param["vW"] = mat_add(rescale(element_wise_mul(param["dW"], param["dW"]), 1 - self.beta2), rescale(param["vW"], self.beta2))
            mw_hat = rescale(param["mW"],  1 / (1 - self.beta1 ** self.k))
            vw_hat = rescale(param["vW"], 1 / (1 - self.beta2 ** self.k))
            grad_step_w = element_wise_mul(mw_hat, element_wise_rev(add_scalar(mat_sqrt(vw_hat), self.eps)))
            param["W"] = mat_add(param["W"], rescale(grad_step_w, -self.lr))

            param["mb"] = mat_add(rescale(param["db"], 1 - self.beta1), rescale(param["mb"], self.beta1))
            param["vb"] = mat_add(rescale(element_wise_mul(param["db"], param["db"]), 1 - self.beta2), rescale(param["vb"], self.beta2))
            mb_hat = rescale(param["mb"], 1 / (1 - self.beta1 ** self.k))
            vb_hat = rescale(param["vb"], 1 / (1 - self.beta2 ** self.k))
            grad_step_b = element_wise_mul(mb_hat, element_wise_rev(add_scalar(mat_sqrt(vb_hat), self.eps)))
            param["b"] = mat_add(param["b"], rescale(grad_step_b, -self.lr))
        self.k += 1
