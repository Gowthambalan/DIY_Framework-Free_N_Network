"""Define utilities to be shared across the project and be reachable by different modules.

There are various helper functions and utilities that are used within a Neural Network framework and their usages are
as broad as to be used for example by a Loss function (i.e. MSE) or Neural Net layer (like Conv2d).

It's a common idea among developers to build up a separate module to contain these utilities and consequently, provide
ease-of-use and further, shared accessibility for other out-of-scope modules.

Routine Listings
---------------
check_shapes

binary_cross_entropy

im2col_indices

col2im

conv_out_size

"""
import numpy as np
import inspect
from typing import Callable


def check_shapes(func):
    """Decorate 2-input-argument functions that their arguments must have the same numpy shape.

    It's very common in calculations involved in Neural Networks' implementation and learning loops
    to compute quantities that are based on 2 separate inputs and those inputs must possess the same shape, if one
    is hoping to obtain correct outputs.

    This function acts as a decorator and guarantees that any pair of inputs to functions decorated by it,
    does not violate the presumption of having equal shapes.

    Parameters
    ----------
    func : Callable
        The function to be decorated

    Returns
    -------
    Callable
        The decorated function.

    Notes
    -----
    Class methods, in addition to regular functions, are also supported. In other words,
    you can decorate class methods that one of the arguments is `self` and, this decorator
    will ignore `self` and performs the dimensionality checking on the remianing two arguments.

    Examples
    --------
    **Regular functions**

    >>> @check_shapes
    >>> def binary_cross_entropy(p, t):
    ...     eps = 1e-6
    ...     return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)
    >>> t = np.array([[1, 0, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> binary_cross_entropy(p, t)
    AssertionError: Inputs to the function are in different shapes: (1, 4) and (1, 3) at binary_cross_entropy!

    **Class methods**

    >>> class DummyClass:
    ...    @check_shapes
    ...    def binary_cross_entropy(self, p, t):
    ...        eps = 1e-6
    ...        return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)
    >>> dummy_var = DummyClass()
    >>> t = np.array([[1, 0, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> dummy_var.binary_cross_entropy(p, t)
    AssertionError: Inputs to the function are in different shapes: (1, 4) and (1, 3) at DummyClass.binary_cross_entropy!
    """
    if "self" in inspect.signature(func).parameters:
        def inner_func(self, x, y):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"
            return func(self, x, y)
    else:
        def inner_func(x, y):
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            if not isinstance(y, np.ndarray):
                y = np.array(y)

            assert x.shape == y.shape, \
                f"Inputs to the function are in different shapes: {x.shape} and {y.shape} at {func.__qualname__}!"

            return func(x, y)

    return inner_func


@check_shapes
def binary_cross_entropy(p, t):
    r"""Calculate Binary Cross Entropy.

    When Binary Cross Entropy quantity is needed, this function can be invoked as a wrapper of calculating BCE.

    Parameters
    ----------
    p : array_like
        Prediction probabilities.
    t : array_like
        Target labels.

    Returns
    -------
    numpy.ndarray
        BCE values.

    Raises
    ------
    AssertionError
        If `p` and `t` shapes are not the same.

    Notes
    -----
    Binary Cross Entropy is a special case of cross entropy quantity that is concerned for only 2 categories:

    .. math:: BCE(p, t) = t * \log{(p)} + (1 - t) * \log{(1 - p)}

    `eps=1e-6` is used to stablized the logrithm function when it encounters 0 as its input.

    Examples
    --------
    >>> t = np.array([[1, 0, 1, 1]])
    >>> p = np.array([[0.85, 0.2, 0.5, 0.95]])
    >>> binary_cross_entropy(p, t)
    [[-0.16251775 -0.2231423  -0.69314518 -0.05129224]]
    """
    eps = 1e-6
    return t * np.log(p + eps) + (1 - t) * np.log(1 - p + eps)


def im2col_indices(x, kernel_size, stride, padding):  # noqa
    r"""Find image-to-column transformation indices in the 3D input array.

    To perform tensor operations (like Convolution) on 3D inputs (like images) in Deep Learning, naive approaches
    are extremely inefficient (i.e. convolution operation on an image is at least in O(n^3)).

    One idea that reduces the time complexity (but is quite memory-intensive) is that to transform 3D inputs to 2D
    matrices and treat the operations involved like regular matrix-dot-products that we do in Fully-Connected layers and
    , consequently lowering the time complexity.

    Image-to-Column method computes coordinates of patches that are going to be met by the convolution operation
    across all input channels and returns those coordinates thus, one can transform the 3D input array to a resulting
    2D matrix consisting only the patches from the original array that are going to be met by the convolution filter.

    Parameters
    ----------
    x : numpy.ndarray
        Input 3D array in shape (N, rows, cols, channels).
    kernel_size : tuple of int or list of int
        Kernel size of the filter.
    stride : int
        Stride in both x and y directions.
    padding : tuple of int or list of int
        Amount of padding applied to rows and columns

    Returns
    -------
    i : numpy.ndarray
        Indices over row axis.
    j : numpy.ndarray
        Indices over column axis.
    k : numpy.ndarray
        Indices over depth axis.

    References
    ----------
    .. [1] `CS231n (2016), Lecture 11, "CNNs in practice,"
       <http://cs231n.stanford.edu/slides/2016/winter1516_lecture11.pdf>`_

    .. [2] `Warden (2015). "Why GEMM is at the heart of deep learning,"
       <https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/>`_

    .. [3] `Chouri (2019). "Demystifying the math and implementation of Convolutions: Part III,"
        <https://praisethemoon.org/demystifying-the-math-and-implementation-of-convolutions-part-iii/>`_

    .. [4] `numpy-ml
        <https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L447>`_

    .. [5] `Kazemipour (2022).
        "img2col_explained" <https://gist.github.com/alirezakazemipour/745041bbcdd294ad5d2049cc975f64aa>`_

    Examples
    --------
    >>> x = np.random.randint(0, 3, size=(1, 5, 5, 1))
    >>> i, j, k = im2col_indices(x, kernel_size=(1, 1), stride=2, padding=(0, 0))
    >>> x[:, i, j, k]
    [[[0]
      [2]
      [2]
      [0]
      [1]
      [1]
      [2]
      [0]
      [0]]]
    """
    fr, fc = kernel_size
    pr, pc = padding
    batch_size, in_rows, in_cols, in_channel = x.shape

    out_rows = conv_out_size(in_rows, fr, stride, pr)
    out_cols = conv_out_size(in_cols, fc, stride, pc)

    i0 = np.repeat(np.arange(fr), fc)
    i0 = np.tile(i0, in_channel)
    i1 = stride * np.repeat(np.arange(out_rows), out_cols)
    j0 = np.tile(np.arange(fc), fr * in_channel)
    j1 = stride * np.tile(np.arange(out_cols), out_rows)

    # i.shape = (out_height * out_width, k^2C)
    # j.shape = (out_height * out_width, k^2C)
    # k.shape = (1, k^2C)
    i = i0.reshape(1, -1) + i1.reshape(-1, 1)
    j = j0.reshape(1, -1) + j1.reshape(-1, 1)
    k = np.repeat(np.arange(in_channel), fr * fc).reshape(1, -1)
    return i, j, k


def col2im(x_col, i, j, k, batch_size, n_rows, n_cols, n_channel, kernel_size, padding):
    """Transform a 2D matrix --generated by the *im2col* transformation-- to its corresponding 3D array.

    Based on the indices chosen by the im2col procedure, col2im can produce the corresponding 3D array.

    Parameters
    ----------
    x_col : numpy.ndarray
        2D input matrix.
    i : numpy.ndarray
        Indices over row axis generated by im2col routine.
    j : numpy.ndarray
        Indices over column axis generated by im2col routine.
    k : numpy.ndarray
        Indices over depth axis generated by im2col routine.
    batch_size : int
        Batch size of the fed data.
    n_rows : int
        Number of input's rows.
    n_cols : int
        Number of input's columns.
    n_channel : int
        Number of input's channels.
    kernel_size : tuple of int or list of int
        Kernel size of the filter.
    padding : tuple of int or list of int
        Amount of padding applied to rows and columns

    Returns
    -------
    numpy.ndarray
        3D array corresponding to the 2D input matrix that had been generated by im2col.

    See Also
    --------
    im2col_indices : Returns indices used for image-to-column transformation.

    References
    ----------
    .. [1] `numpy-ml
        <https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/neural_nets/utils/utils.py#L546>`_

    Examples
    --------
    >>> x = np.random.randint(0, 3, size=(1, 5, 5, 1))
    >>> i, j, k = im2col_indices(x, kernel_size=(1, 1), stride=1, padding=(0, 0))
    >>> x_col = x[:, i, j, k]
    >>> y = col2im(x_col, i=i, j=j, k=k, batch_size=1, n_rows=5, n_cols=5, n_channel=1, kernel_size=(1, 1), padding=(0, 0))
    >>> print(f"x = {x[0, :, :, 0]}")
    >>> print(f"y = {y[0, :, :, 0].astype(int)}")
    x = [[0 0 1 1 1]
    [2 1 2 0 2]
    [1 0 2 2 1]
    [1 1 0 1 2]
    [0 0 2 2 1]]
    y = [[0 0 1 1 1]
    [2 1 2 0 2]
    [1 0 2 2 1]
    [1 1 0 1 2]
    [0 0 2 2 1]]
    """
    pr, pc = padding
    fr, fc = kernel_size
    x_pad = np.zeros((batch_size, n_rows + 2 * pr, n_cols + 2 * pc, n_channel))

    x_col_reshaped = x_col.reshape(-1, n_channel * fr * fc, batch_size)
    x_col_reshaped = x_col_reshaped.transpose(2, 0, 1)

    np.add.at(x_pad, (slice(None), i, j, k), x_col_reshaped)

    pr2 = None if pr == 0 else -pr
    pc2 = None if pc == 0 else -pc
    return x_pad[:, pr:pr2, pc:pc2, :]


def conv_out_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate the output size of convolution operation on the input.

    Parameters
    ----------
    input_size : int
        Row/Column size of the input tensor.
    kernel_size : int
        Kernel size of the convolution filter.
    stride : int, default 1
        Stride of the convolution filter.
    padding : int, default 0
        The amount of padding added to the input's given dimension.

    Returns
    -------
    int
        Output size of the convolution operation on the given input's dimension.

    Notes
    -----
    .. math:: n_{out} = \lfloor\\frac{n_{in} + 2 * p - k}{s}\\rfloor + 1

    Examples
    --------
    >>> x = np.random.random((5, 5))
    >>> conv_out_size(x.shape[0], kernel_size=2, stride=2, padding=1)
    3
    """
    return (input_size + 2 * padding - kernel_size) // stride + 1


if __name__ == "__main__":
    x = np.random.randint(0, 3, size=(1, 5, 5, 1))
    i, j, k = im2col_indices(x, kernel_size=(1, 1), stride=1, padding=(0, 0))
    x_col = x[:, i, j, k]
    y = col2im(x_col, i=i, j=j, k=k, batch_size=1, n_rows=5, n_cols=5, n_channel=1, kernel_size=(1, 1), padding=(0, 0))
    print(f"x = {x[0, :, :, 0]}")
    print(f"y = {y[0, :, :, 0].astype(int)}")
