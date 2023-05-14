import collections

import calc
import numpy as np


def transpose(vector):
    d = len(vector)
    ret = np.zeros((d, 1))
    for i in range(d):
        ret[i][0] = vector[i]
    return np.array(ret)


def get_v(s, y, ro, dim):
    return np.identity(dim) - ro * np.outer(s, y)


def get_u(ro, s):
    return ro * np.outer(s, s)


def update(s, y, matrix_h, ro, dim):
    matrix_v = get_v(s=s, y=y, ro=ro, dim=dim)
    matrix_v_t = np.transpose(matrix_v)
    matrix_u = get_u(ro=ro, s=s)
    return np.dot(np.dot(matrix_v_t, matrix_h), matrix_v) + matrix_u


def l_bfgs(f, start_point, epoch=10000,  m=10):
    grad_eps = 1e-7
    x = start_point
    g = calc.gradient(f, x)
    dim = len(start_point)
    points = np.zeros((epoch + 1, dim))
    points[0] = start_point
    last_updates_s = collections.deque()
    last_updates_y = collections.deque()
    last_updates_ro = collections.deque()
    alpha = np.zeros(m)
    beta = np.zeros(m)
    z = 0.018 * g
    for k in range(1, epoch+1):
        x = x - z
        points[k] = x
        g_1 = calc.gradient(f, x)
        s = points[k] - points[k - 1]
        y = g_1 - g
        ro = 1 / (np.dot(y, s))
        g = g_1
        if np.linalg.norm(g) < grad_eps:
            print("Потребовалось итераций " + str(k))
            break

        if k > m:
            last_updates_s.popleft()
            last_updates_y.popleft()
            last_updates_ro.popleft()
        last_updates_s.append(s)
        last_updates_y.append(y)
        last_updates_ro.append(ro)
        q = g
        size = len(last_updates_s)
        for i in range(size - 1, -1, -1):
            alpha[i] = last_updates_ro[i] * np.dot(last_updates_s[i], q)
            q = q - alpha[i] * last_updates_y[i]
        gamma = np.dot(last_updates_s[size-1], last_updates_y[size-1]) / np.dot(last_updates_y[size - 1],
                                                                                last_updates_y[size - 1])
        matrix_h = gamma * np.identity(dim)
        z = np.dot(matrix_h, q)
        for i in range(size):
            beta[i] = last_updates_ro[i] * np.dot(last_updates_y[i], z)
            z = z + last_updates_s[i] * (alpha[i] - beta[i])

    return x
