import numpy as np

from .utils import *
from .activations import *
from .initializers import *


def supported_layers():
    return [x.__name__ for x in ParamLayer.__subclasses__()]


class Layer:
    def __init__(self):
        self.vars = {}

    def forward(self, x):
        raise NotImplementedError

    def backward(self, x):
        raise NotImplementedError


class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer,
                 regularizer_type: str = None,
                 lam: float = 0.):
        super().__init__()

        i, j = weight_shape
        self.vars["W"] = weight_initializer.initialize([[0 for _ in range(j)] for _ in range(i)])
        self.vars["b"] = [bias_initializer.initialize([0 for _ in range(j)])]
        self.vars["dW"] = [[0 for _ in range(j)] for _ in range(i)]
        self.vars["db"] = [[0 for _ in range(j)]]

        self.z = None
        self.input = None

        self.regularizer_type = regularizer_type
        self.lam = lam


class Dense(ParamLayer, ABC):
    def __init__(self, in_features: int,
                 out_features: int,
                 activation: Activation = Linear(),
                 weight_initializer: Initializer = RandomUniform(),
                 bias_initializer: Initializer = Constant(),
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        super().__init__(weight_shape=(in_features, out_features),
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lam=lam
                         )
        self.in_features = in_features
        self.out_features = out_features
        self.act = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = np.ndarray.tolist(x)
        assert isinstance(x, list)
        assert isinstance(x[0], list), "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        # z = x.dot(self.vars["W"]) + self.vars["b"]
        z = mat_mul(x, self.vars["W"])
        b = deepcopy(self.vars["b"])
        while len(b) < len(x):
            b.append(self.vars["b"][0])
        z = mat_add(z, b)
        self.z = z
        a = self.act(z)
        return a

    def backward(self, delta):
        dz = element_wise_mul(delta, self.act.derivative(self.z))
        input_t = transpose(self.input)
        dw_unscale = mat_mul(input_t, dz)
        self.vars["dW"] = rescale(dw_unscale, 1 / len(dz))
        # self.vars["db"] = np.sum(dz, axis=0) / dz.shape[0]
        ones_t = [[1 for _ in range(len(dz))] for _ in range(1)]
        db_unscale = mat_mul(ones_t, dz)
        self.vars["db"] = rescale(db_unscale, 1 / len(dz))

        if self.regularizer_type == "l2":
            self.vars["dW"] = mat_add(self.vars["dW"], rescale(self.vars["W"], self.lam))
            # self.vars["db"] = mat_add(self.vars["db"], rescale(self.vars["b"], self.lam))

        elif self.regularizer_type == "l1":
            self.vars["dW"] = add_scalar(self.vars["dW"], self.lam)
            # self.vars["db"] = add_scalar(self.vars["db"], self.lam)

        w_t = transpose(self.vars["W"])
        # delta = dz.dot(self.vars["W"].T)
        delta = mat_mul(dz, w_t)
        return delta

    def __call__(self, x):
        return self.forward(x)


class BatchNorm1d(ParamLayer, ABC):
    #  https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
    def __init__(self, in_features: int):
        super().__init__(weight_shape=(1, in_features),
                         weight_initializer=Constant(1.),
                         bias_initializer=Constant(0.)
                         )
        self.in_features = in_features
        self.x_hat = None
        self.eps = 1e-5
        self.beta = 0.1
        self.mu = 0
        self.std = 0
        self.mu_hat = [[0 for _ in range(self.in_features)]]
        self.std_hat = [[1 for _ in range(self.in_features)]]
        self.gamma = None

    def forward(self, x, eval=False):
        assert isinstance(x, list)
        assert isinstance(x[0], list), "Feed the input to the network in batch mode: (batch_size, n_dims)"
        if not eval:
            self.mu = batch_mean(x)
            self.std = mat_sqrt(batch_var(x, self.mu))
            self.mu_hat = mat_add(rescale(self.mu_hat, 1 - self.beta), rescale(self.mu, self.beta))
            self.std_hat = mat_add(rescale(self.std_hat, 1 - self.beta), rescale(self.std, self.beta))
        else:
            self.mu = self.mu_hat
            self.std = self.std_hat
        mu = deepcopy(self.mu)
        std = deepcopy(self.std)
        while len(mu) < len(x):
            mu.append(self.mu[0])
            std.append(self.std[0])
        num = mat_add(x, rescale(mu, -1))
        den = mat_sqrt(add_scalar(element_wise_mul(std, std), self.eps))
        x_hat = element_wise_mul(num, element_wise_rev(den))
        self.x_hat = x_hat

        self.gamma = deepcopy(self.vars["W"])
        beta = deepcopy(self.vars["b"])
        while len(self.gamma) < len(x):
            self.gamma.append(self.vars["W"][0])
            beta.append(self.vars["b"][0])

        y = mat_add(element_wise_mul(self.gamma, x_hat), beta)
        return y

    def backward(self, delta):
        #  https://kevinzakka.github.io/2016/09/14/batch_normalization/
        dz = delta
        dx_hat = element_wise_mul(dz, self.gamma)
        m = len(dz)
        self.vars["dW"] = rescale(batch_sum(element_wise_mul(self.x_hat, dz)), 1 / m)
        self.vars["db"] = rescale(batch_sum(dz), 1 / m)

        a1 = rescale(dx_hat, m)
        a2 = batch_sum(dx_hat)
        a3 = element_wise_mul(*equal_batch_size(self.x_hat, batch_sum(element_wise_mul(dx_hat, self.x_hat))))
        num = mat_add(a1, mat_add(*equal_batch_size(rescale(a2, -1), rescale(a3, -1))))
        den = rescale(mat_sqrt(add_scalar(element_wise_mul(self.std, self.std), self.eps)), m)

        delta = element_wise_mul(*equal_batch_size(num, element_wise_rev(den)))
        return delta

    def __call__(self, x, eval=False):
        return self.forward(x, eval)
