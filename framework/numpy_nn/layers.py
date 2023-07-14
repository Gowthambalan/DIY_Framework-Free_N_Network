from collections import namedtuple
from .initializers import *
from .activations import *
from .utils import *
from typing import Optional


def supported_layers():
    return [x.__name__ for x in ParamLayer.__subclasses__()] # noqa


# region Layer
class Layer:
    def forward(self, **kwargs):
        raise NotImplementedError

    def backward(self, **x):
        raise NotImplementedError

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


# endregion

# region ParamLayer
class ParamLayer(Layer, ABC):
    def __init__(self,
                 weight_shape,
                 weight_initializer,
                 bias_initializer,
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        self.vars = {"W": weight_initializer.initialize(weight_shape),
                     "b": bias_initializer.initialize((1, weight_shape[1])),
                     "dW": np.zeros(weight_shape),
                     "db": np.zeros((1, weight_shape[1]))}

        self.z = None
        self.input = None

        self.regularizer_type = regularizer_type  # noqa
        self.lam = lam

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1])
        return name, output_shape, n_param

    @property
    def input_shape(self):
        return self.vars["W"].shape[0]


# endregion

# region Dense
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

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        self.input = x
        z = x.dot(self.vars["W"]) + self.vars["b"]
        self.z = z
        a = self.act(z)
        return a

    def backward(self, **delta):
        #  https://cs182sp21.github.io/static/slides/lec-5.pdf
        delta = delta["delta"]
        dz = delta * self.act.derivative(self.z)
        self.vars["dW"] = self.input.T.dot(dz) / dz.shape[0]
        self.vars["db"] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.regularizer_type == "l2":
            self.vars["dW"] += self.lam * self.vars["W"]
            # Biases are not regularized: https://cs231n.github.io/neural-networks-2/#reg
            # self.vars["db"] += self.lam * self.vars["b"]
        elif self.regularizer_type == "l1":
            self.vars["dW"] += self.lam
            # self.vars["db"] += self.lam

        delta = dz.dot(self.vars["W"].T)
        return dict(delta=delta)

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


# endregion

# region BatchNorm1d
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
        self.mu_hat = 0
        self.std_hat = 0

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        if not eval:
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.std = np.std(x, axis=0, keepdims=True)
            self.mu_hat = (1 - self.beta) * self.mu_hat + self.beta * self.mu
            self.std_hat = (1 - self.beta) * self.std_hat + self.beta * self.std
        else:
            self.mu = self.mu_hat
            self.std = self.std_hat
        x_hat = (x - self.mu) / np.sqrt(self.std ** 2 + self.eps)
        self.x_hat = x_hat
        y = self.vars["W"] * x_hat + self.vars["b"]
        return y

    def backward(self, **delta):
        #  https://kevinzakka.github.io/2016/09/14/batch_normalization/
        delta = delta["delta"]
        dz = delta
        dx_hat = dz * self.vars["W"]
        m = dz.shape[0]
        self.vars["dW"] = np.sum(self.x_hat * dz, axis=0) / m
        self.vars["db"] = np.sum(dz, axis=0) / m

        delta = (m * dx_hat - np.sum(dx_hat, axis=0, keepdims=True) - self.x_hat * np.sum(
            dx_hat * self.x_hat, axis=0, keepdims=True)) / (m * np.sqrt(self.std ** 2 + self.eps))
        return dict(delta=delta)

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


# endregion

# region Dropout
class Dropout(Layer, ABC):
    """
    - References:
        1. https://cs231n.github.io/neural-networks-2/#reg
        2. https://deepnotes.io/dropout
    """

    def __init__(self, p: float = 0.5):
        """
        :param p: float
        Probability of keeping a neuron active.
        """

        self.p = 1 - p
        self.mask = None

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        if not eval:
            self.mask = (np.random.rand(*x.shape) < self.p) / self.p
            return x * self.mask
        else:
            return x

    def backward(self, **delta):
        delta = delta["delta"]
        return dict(delta=delta * self.mask)

    def summary(self):
        raise NotImplementedError

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


# endregion

# region LSTMCell
class LSTMCell(ParamLayer, ABC):
    """
    - References:
        1. https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/layers/layers.py
        2. http://arunmallya.github.io/writeups/nn/lstm/index.html#/
        3. http://cs231n.stanford.edu/slides/2019/cs231n_2019_lecture10.pdf
    """

    def __init__(self, in_features: int,
                 hidden_size: int,
                 weight_initializer: Initializer = RandomUniform(),
                 bias_initializer: Initializer = Constant(),
                 regularizer_type: str = None,  # noqa
                 lam: float = 0.
                 ):
        weight_shape = (in_features + hidden_size, 4 * hidden_size)
        super().__init__(weight_shape=weight_shape,
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lam=lam
                         )
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam

        self.Buffer = namedtuple("Buffer", ["input", "ct", "ot", "gt", "it", "ct_1", "ft", "gt_hat", "dht_1", "dct_1"])
        self.t = 0
        self.buffer = {}

    def forward(self, x, ht_1, ct_1, eval=False):
        self.buffer[self.t] = self.Buffer(None, None, None, None, None, None, None, None, None, None)
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        input = np.hstack([x, ht_1])
        self.buffer[self.t] = self.buffer[self.t]._replace(input=input)
        self.buffer[self.t] = self.buffer[self.t]._replace(ct_1=ct_1)
        zt = input.dot(self.vars["W"]) + self.vars["b"]
        it_hat, ft_hat, ot_hat, gt_hat = np.split(zt, 4, axis=-1)
        self.buffer[self.t] = self.buffer[self.t]._replace(gt_hat=gt_hat)
        it = Sigmoid.forward(it_hat)
        self.buffer[self.t] = self.buffer[self.t]._replace(it=it)
        ft = Sigmoid.forward(ft_hat)
        self.buffer[self.t] = self.buffer[self.t]._replace(ft=ft)
        ot = Sigmoid.forward(ot_hat)
        self.buffer[self.t] = self.buffer[self.t]._replace(ot=ot)
        gt = np.tanh(gt_hat)
        self.buffer[self.t] = self.buffer[self.t]._replace(gt=gt)
        ct = ft * ct_1 + it * gt
        self.buffer[self.t] = self.buffer[self.t]._replace(ct=ct)
        ht = ot * np.tanh(ct)
        return ht, ct

    def backward(self, **delta):
        dht = delta.get("dht", 0) + delta["delta"]
        dct = delta.get("dct", np.zeros_like(dht))
        ct = self.buffer[self.t].ct
        do = dht * np.tanh(ct)
        ot = self.buffer[self.t].ot
        dct += dht * ot * (1 - np.tanh(ct) ** 2)
        gt = self.buffer[self.t].gt
        dit = dct * gt
        ct_1 = self.buffer[self.t].ct_1
        dft = dct * ct_1
        it = self.buffer[self.t].it
        dgt = dct * it
        ft = self.buffer[self.t].ft
        dct_1 = dct * ft
        self.buffer[self.t] = self.buffer[self.t]._replace(dct_1=dct_1)
        gt_hat = self.buffer[self.t].gt_hat
        dgt_hat = dgt * (1 - np.tanh(gt_hat) ** 2)
        dit_hat = dit * it * (1 - it)
        dft_hat = dft * ft * (1 - ft)
        dot_hat = do * ot * (1 - ot)
        dzt = np.hstack([dit_hat, dft_hat, dot_hat, dgt_hat])

        input = self.buffer[self.t].input
        self.vars["dW"] = input.T.dot(dzt) / dzt.shape[0]
        self.vars["db"] = np.sum(dzt, axis=0, keepdims=True) / dzt.shape[0]

        if self.regularizer_type == "l2":
            self.vars["dW"] += self.lam * self.vars["W"]
        elif self.regularizer_type == "l1":
            self.vars["dW"] += self.lam

        dinput = dzt.dot(self.vars["W"].T)
        dx, dht_1 = np.split(dinput, [self.in_features], axis=-1)
        self.buffer[self.t] = self.buffer[self.t]._replace(dht_1=dht_1)

        return dict(delta=dx)

    def summary(self):
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.vars["b"].shape[1] // 4)
        return name, output_shape, n_param

    def __call__(self, x, h, c, eval=False):
        return self.forward(x, h, c, eval)


# endregion

# region LSTM
class LSTM(LSTMCell, ABC):
    """
    - References:
        1. https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/layers/layers.py
        2. http://arunmallya.github.io/writeups/nn/lstm/index.html#/
    """

    def __init__(self, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.seq_len = None
        self.batch_size = None

    def forward(self, x, h, c, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) == 3, "x should be in (batch_size, sequence_length, in_features) shape!"
        output = []
        self.batch_size, self.seq_len, num_feats = x.shape
        for t in range(self.seq_len):
            self.t = t
            h, c = super(LSTM, self).forward(x[:, t, :], h, c, eval)
            output.append(h)
        return np.stack(output, axis=-2), h, c

    def backward(self, **delta):
        if len(delta["delta"].shape) != 3:
            tmp = np.zeros((self.batch_size, self.seq_len - 1, self.hidden_size))
            da = np.expand_dims(delta["delta"], axis=1)
            da = np.concatenate([tmp, da], axis=1)
        else:
            da = delta["delta"]

        dW, db = np.array(0), np.array(0)
        dx = np.zeros((self.batch_size, self.seq_len, self.in_features))
        for t in reversed(range(self.seq_len)):
            self.t = t
            delta["delta"] = da[:, t, :]
            delta = super(LSTM, self).backward(**delta)
            delta["dht"] = self.buffer[self.t].dht_1
            delta["dct"] = self.buffer[self.t].dct_1
            dW += self.vars["dW"]
            db += self.vars["db"]
            dx[:, t, :] = delta["delta"]

        self.vars["dW"] = dW
        self.vars["db"] = db
        delta.pop("dht")
        delta.pop("dct")
        return dict(delta=dx)

    def __call__(self, x, h, c, eval=False):
        return self.forward(x, h, c, eval)

    @property
    def input_shape(self):
        return self.seq_len, self.in_features


# endregion

# region Conv2d
class Conv2d(ParamLayer, ABC):
    def __init__(self, in_features: int,
                 out_features: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 activation: Activation = Linear(),
                 weight_initializer: Initializer = RandomUniform(),
                 bias_initializer: Initializer = Constant(),
                 regularizer_type: str = None,
                 lam: float = 0.
                 ):
        self.in_features = in_features
        self.out_features = out_features
        self.act = activation
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam
        self.pad_width = (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)
        self.in_rows = None
        self.in_cols = None
        self.batch_size = None
        self.i, self.j, self.k = None, None, None

        super().__init__(weight_shape=(self.kernel_size[0] * self.kernel_size[1] * in_features, out_features),
                         weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer,
                         regularizer_type=regularizer_type,
                         lam=lam
                         )

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        assert len(x.shape) == 4, f"Invalid input shape in {self.__class__.__name__}." \
                                  f" Valid shape is: (batch_size, n_rows, n_cols, in_features)"

        self.in_rows = x.shape[1]
        self.in_cols = x.shape[2]
        self.batch_size = x.shape[0]
        x_pad = np.pad(x,
                       pad_width=self.pad_width,
                       mode="constant",
                       constant_values=0,
                       )
        i, j, k = im2col_indices(x, self.kernel_size, self.stride, self.padding)
        x_col = x_pad[:, i, j, k]
        self.i, self.j, self.k = i, j, k
        self.input = x_col
        z = x_col.dot(self.vars["W"]) + self.vars["b"]
        self.z = z
        a = self.act(z)
        out_rows = conv_out_size(self.in_rows, self.kernel_size[0], self.stride, self.padding[0])
        out_cols = conv_out_size(self.in_cols, self.kernel_size[1], self.stride, self.padding[1])
        return a.reshape((self.batch_size, out_rows, out_cols, self.out_features))

    def backward(self, **delta):
        #  https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/layers/layers.py#L3048
        delta = delta["delta"].reshape((-1, self.out_features))
        z = self.z.reshape(delta.shape)
        dz = delta * self.act.derivative(z)
        input = self.input.reshape((dz.shape[0], -1))
        self.vars["dW"] = input.T.dot(dz) / dz.shape[0]
        self.vars["db"] = np.sum(dz, axis=0, keepdims=True) / dz.shape[0]

        if self.regularizer_type == "l2":
            self.vars["dW"] += self.lam * self.vars["W"]
        elif self.regularizer_type == "l1":
            self.vars["dW"] += self.lam

        delta = dz.dot(self.vars["W"].T)
        delta = col2im(delta, self.i, self.j, self.k,
                       self.batch_size, self.in_rows,
                       self.in_cols, self.in_features,
                       self.kernel_size, self.padding
                       )
        return dict(delta=delta)

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


# endregion

# region Conv1d
class Conv1d(Conv2d, ABC):
    def __init__(self, seq_len: Optional = None, **kwargs):
        kwargs["kernel_size"] = 1, kwargs["kernel_size"]
        kwargs["padding"] = 0, kwargs["padding"]
        self.seq_len = seq_len
        super().__init__(**kwargs)

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        assert len(x.shape) == 3, f"Invalid input shape in {self.__class__.__name__}." \
                                  f" Valid shape is: (batch_size, seq_len, in_features)"
        x = np.expand_dims(x, axis=1)
        a = super().forward(x)
        a = np.squeeze(a, axis=1)
        return a

    def backward(self, **delta):
        delta = super().backward(**delta)
        delta["delta"] = np.squeeze(delta["delta"], axis=1)
        return delta

    def summary(self):
        if self.seq_len is None:
            raise AttributeError(
                f"`seq_len` should be specified prior to invoking summary! in {self.__class__.__name__}")
        name = self.__class__.__name__
        n_param = self.vars["W"].shape[0] * self.vars["W"].shape[1] + self.vars["b"].shape[1]
        output_shape = (None, self.seq_len, self.vars["b"].shape[1])
        return name, output_shape, n_param

    @property
    def input_shape(self):
        return self.in_cols if self.in_cols is not None else self.seq_len, self.in_features


# endregion

# region LayerNorm
class LayerNorm(ParamLayer, ABC):
    # https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/layers/layers.py#L1634
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

    def forward(self, x, eval=False):  # noqa
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        feat_dims = tuple(range(x.ndim))[1:]
        self.mu = np.mean(x, axis=feat_dims, keepdims=True)
        self.std = np.std(x, axis=feat_dims, keepdims=True)

        x_hat = (x - self.mu) / np.sqrt(self.std ** 2 + self.eps)
        self.x_hat = x_hat
        y = self.vars["W"] * x_hat + self.vars["b"]
        return y

    def backward(self, **delta):
        delta = delta["delta"]
        dz = delta
        dx_hat = dz * self.vars["W"]
        m = dz.shape[0]
        self.vars["dW"] = np.sum(self.x_hat * dz, axis=0) / m
        self.vars["db"] = np.sum(dz, axis=0) / m

        n_dims = np.prod(dz.shape[1:])
        feat_dims = tuple(range(dz.ndim))[1:]
        delta = (n_dims * dx_hat - np.sum(dx_hat, axis=feat_dims, keepdims=True) - self.x_hat * np.sum(
            dx_hat * self.x_hat, axis=feat_dims, keepdims=True)) / (n_dims * np.sqrt(self.std ** 2 + self.eps))
        return dict(delta=delta)

    def __call__(self, x, eval=False):  # noqa
        return self.forward(x, eval)


# endregion

# region Pool2d
class Pool2d(Layer, ABC):
    # https://github.com/madalinabuzau/cs231n-convolutional-neural-networks-solutions/blob/master/2017%20Spring%20Assignments/assignment2/cs231n/fast_layers.py
    # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    def __init__(self,
                 mode: str,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 ):
        self.mode = mode
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.pad_width = (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0)
        self.in_rows = None
        self.in_cols = None
        self.in_channels = None
        self.batch_size = None
        self.i, self.j, self.k = None, None, None
        self.input = None
        self.idx = None

    def forward(self, x, eval=False):  # noqa
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        assert len(x.shape) == 4, f"Invalid input shape in {self.__class__.__name__}." \
                                  f" Valid shape is: (batch_size, n_rows, n_cols, in_features)"

        self.batch_size = x.shape[0]
        self.in_rows = x.shape[1]
        self.in_cols = x.shape[2]
        self.in_channels = x.shape[3]

        x_single_channel = x.reshape((self.batch_size * self.in_channels, self.in_rows, self.in_cols, 1))
        i, j, k = im2col_indices(x_single_channel, self.kernel_size, self.stride, self.padding)
        if self.mode == "max":
            x_pad = np.pad(x_single_channel,
                           pad_width=self.pad_width,
                           mode="constant",
                           constant_values=-np.inf,
                           )
        elif self.mode == "avg":
            x_pad = np.pad(x_single_channel,
                           pad_width=self.pad_width,
                           mode="constant",
                           constant_values=0,
                           )
        else:
            raise ValueError(f"Invalid mode of pooling: {self.mode}. Available modes ('max', 'avg').")

        x_col = x_pad[:, i, j, k]
        self.i, self.j, self.k = i, j, k
        self.input = x_col
        if self.mode == "max":
            self.idx = np.argmax(x_col, axis=-1)
            cols_idx = np.tile(np.arange(x_col.shape[1]), (self.batch_size * self.in_channels, 1))
            batch_idx = np.arange(self.batch_size * self.in_channels)[:, np.newaxis]
            out = x_col[batch_idx, cols_idx, self.idx]
        elif self.mode == "avg":
            out = np.mean(x_col, axis=-1)

        out_rows = conv_out_size(self.in_rows, self.kernel_size[0], self.stride, self.padding[0])
        out_cols = conv_out_size(self.in_cols, self.kernel_size[1], self.stride, self.padding[1])
        return out.reshape((self.batch_size, out_rows, out_cols, self.in_channels))  # noqa

    def backward(self, **delta):
        dout = delta["delta"].reshape(self.batch_size * self.in_channels, -1)  # noqa
        delta = np.zeros_like(self.input)
        if self.mode == "max":
            cols_idx = np.tile(np.arange(self.input.shape[1]), (self.batch_size * self.in_channels, 1))
            batch_idx = np.arange(self.batch_size * self.in_channels)[:, np.newaxis]
            delta[batch_idx, cols_idx, self.idx] = dout
        elif self.mode == "avg":
            delta = np.tile(dout[..., np.newaxis], (1, 1, self.input.shape[-1])) / self.input.shape[-1]

        delta = col2im(delta,
                       self.i,
                       self.j,
                       self.k,
                       self.batch_size * self.in_channels,
                       self.in_rows,
                       self.in_cols,
                       1,
                       self.kernel_size,
                       self.padding
                       )
        delta = delta.reshape((self.batch_size, self.in_rows, self.in_cols, self.in_channels))
        return dict(delta=delta)

    def summary(self):
        raise NotImplementedError

    def __call__(self, x, eval=False):
        return self.forward(x, eval)


# endregion

# region Pool1d
class Pool1d(Pool2d, ABC):
    def __init__(self, **kwargs):
        kwargs["kernel_size"] = 1, kwargs["kernel_size"]
        kwargs["padding"] = 0, kwargs["padding"]
        super().__init__(**kwargs)

    def forward(self, x, eval=False):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        assert len(x.shape) > 1, "Feed the input to the network in batch mode: (batch_size, n_dims)"
        assert len(x.shape) == 3, f"Invalid input shape in {self.__class__.__name__}." \
                                  f" Valid shape is: (batch_size, seq_len, in_features)"
        x = np.expand_dims(x, axis=1)
        a = super().forward(x)
        a = np.squeeze(a, axis=1)
        return a

    def backward(self, **delta):
        delta = super().backward(**delta)
        delta["delta"] = np.squeeze(delta["delta"], axis=1)
        return delta

    def summary(self):
        raise NotImplementedError

# endregion
