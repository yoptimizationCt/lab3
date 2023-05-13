import numpy as np


def gradient(f, x, h=10e-6):
    grad = np.zeros(len(x))
    delta_x = x.copy()
    for i in range(len(x)):
        delta_x[i] += h
        grad[i] = (f(delta_x) - f(x)) / h
        delta_x[i] -= h
    return grad
