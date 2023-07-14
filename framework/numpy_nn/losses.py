import numpy as np
from .utils import binary_cross_entropy, check_shapes
from abc import ABC


def supported_losses():
    return [x.__name__ for x in LossFunc.__subclasses__()]


class Loss:
    def __init__(self, value, delta):
        self.value = value
        self.delta = delta


class LossFunc:
    def __init__(self, pred=None, target=None):
        self.pred = pred
        self.target = target
        self.eps = 1e-6

    @check_shapes
    def apply(self, p, t):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def __call__(self, p, t):
        return self.apply(p, t)


class MSE(LossFunc, ABC):
    def __init__(self):
        super(MSE, self).__init__()

    @check_shapes
    def apply(self, p, t):
        super(MSE, self).__init__(p, t)
        return Loss(np.mean((p - t) ** 2, axis=0) / 2, self.delta)

    @property
    def delta(self):
        return self.pred - self.target


class CrossEntropy(LossFunc, ABC):
    #  https://cs231n.github.io/neural-networks-case-study/#grad
    def __init__(self):
        super(CrossEntropy, self).__init__()

    @check_shapes
    def apply(self, p, t):
        super(CrossEntropy, self).__init__(p, t)
        probs = self.soft_max(p)
        loss = -np.log(probs[range(p.shape[0]), np.array(t).squeeze(-1)])

        return Loss(np.mean(loss, axis=0), self.delta)

    @property
    def delta(self):
        #  https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
        probs = self.soft_max(self.pred)
        probs[range(self.pred.shape[0]), np.array(self.target).squeeze(-1)] -= 1

        return probs

    @staticmethod
    def soft_max(x):
        logits = x - np.max(x, axis=-1, keepdims=True)
        num = np.exp(logits)
        den = np.sum(num, axis=-1, keepdims=True)
        return num / den


class BinaryCrossEntropy(LossFunc, ABC):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    @check_shapes
    def apply(self, p, t):
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)

        super(BinaryCrossEntropy, self).__init__(p, t)
        loss = -binary_cross_entropy(p, t)
        return Loss(np.mean(loss, axis=0), self.delta)

    @property
    def delta(self):
        return np.expand_dims(self.pred - self.target, -1)


class BinaryFocal(LossFunc, ABC):
    def __init__(self, gamma=2, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        super(BinaryFocal, self).__init__()

    @check_shapes
    def apply(self, p, t):
        if not isinstance(t, np.ndarray):
            t = np.asarray(t)
        if not isinstance(p, np.ndarray):
            p = np.asarray(p)

        super(BinaryFocal, self).__init__(p, t)
        loss = -self.alpha * (1 - p + self.eps) ** self.gamma * binary_cross_entropy(p, t)
        return Loss(np.mean(loss, axis=0), self.delta)

    @property
    def delta(self):
        return np.expand_dims(self.alpha * (1 - self.pred + self.eps) ** (self.gamma - 1) * (
                1 + self.gamma * binary_cross_entropy(self.pred, self.target) -
                self.target / (self.pred + self.eps)), -1)
