import numpy as np
import pickle

from .module import Module
from .sequential import Sequential
from . import activations as acts
from . import initializers as inits
from . import layers
from . import losses
from . import optimizers as optims
from . import utils


def seed(seed):
    np.random.seed(seed)


def save(file: object, name: str):
    with open(name, 'wb') as f:
        pickle.dump(file, f)


def load(name: str):
    with open(name, 'rb') as f:
        return pickle.load(f)
