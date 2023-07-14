import math
from abc import ABC
from .utils import mat_add, rescale


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

    def apply(self, p, t):
        raise NotImplementedError

    @property
    def delta(self):
        raise NotImplementedError

    def __call__(self, p, t):
        return self.apply(p, t)


class MSELoss(LossFunc, ABC):
    def __init__(self):
        super(MSELoss, self).__init__()

    def apply(self, p, t):
        super(MSELoss, self).__init__(p, t)
        # return Loss(np.mean((p - t) ** 2) / 2, self.delta)
        assert isinstance(p, list) and isinstance(t, list)
        assert isinstance(p[0], list) and \
               isinstance(t[0], list), "target and prediction should be in batch mode: (batch_size, n_dims)"

        assert len(p) == len(t) and len(p[0]) == len(t[0])
        loss = 0
        for w, h in zip(p, t):
            for x, y in zip(w, h):
                loss += 0.5 * (x - y) ** 2
        return Loss(loss / len(p), self.delta)

    @property
    def delta(self):
        return mat_add(self.pred, rescale(self.target, -1))


class CrossEntropyLoss(LossFunc, ABC):
    #  https://cs231n.github.io/neural-networks-case-study/#grad
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def apply(self, p, t):
        super(CrossEntropyLoss, self).__init__(p, t)
        assert isinstance(p, list) and isinstance(t, list)
        assert isinstance(p[0], list) and \
               isinstance(t[0], list), "target and prediction should be in batch mode: (batch_size, n_dims)"
        probs = self.soft_max(p)
        w = len(p)
        loss = 0
        for i in range(w):
            loss += -math.log(probs[i][t[i][0]])

        return Loss(loss / w, self.delta)

    @property
    def delta(self):
        #  https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/

        probs = self.soft_max(self.pred)
        w = len(self.pred)
        for i in range(w):
            probs[i][self.target[i][0]] -= 1

        return probs

    @staticmethod
    def soft_max(x):
        w, h = len(x), len(x[0])
        num = [[None for _ in range(h)] for _ in range(w)]
        den = [[None for _ in range(1)] for _ in range(w)]
        for i in range(w):
            max_of_batch = -math.inf
            sum_of_batch = 0
            for j in range(h):
                if x[i][j] > max_of_batch:
                    max_of_batch = x[i][j]
            for j in range(h):
                num[i][j] = math.exp(x[i][j] - max_of_batch)
                sum_of_batch += num[i][j]
            den[i][0] = sum_of_batch

        for i in range(w):
            for j in range(h):
                num[i][j] = num[i][j] / den[i][0] + 1e-6
        return num
