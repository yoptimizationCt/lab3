import numpy as np
from matplotlib import pyplot as plt

from calc import gradient, ternary_search_wolfe
from lab3.paint_contour import paint_contour


def bfgs(f, x0, max_iter=1000, tol=1e-6):
    num_epoch = max_iter
    points = np.zeros((max_iter + 1, 2))
    n = len(x0)
    x = x0
    B = np.eye(n)  # initial approximation of inverse Hessian
    grad = gradient(f, x)
    for epoch in range(max_iter):
        points[epoch] = x
        p = -np.dot(B, grad)
        alpha = ternary_search_wolfe(f, x, p)
        x_new = x + alpha * p
        s = x_new - x
        grad_new = gradient(f, x_new)
        y = grad_new - grad
        if np.linalg.norm(y) < tol:
            points[epoch + 1] = x_new
            num_epoch = epoch
            break
        tt = np.dot(y, s)
        if tt < tol:
            rho = 1e-4
        else:
            rho = 1 / tt
        B = (np.eye(n) - rho * np.outer(s, y)) @ B @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, grad = x_new, grad_new

    return points, num_epoch


f1 = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
f2 = lambda x: 0.01 * x[0] ** 2 + x[1] ** 2
f3 = lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
        2.625 - x[0] + x[0] * x[1] ** 3) ** 2
f4 = lambda x: 0.25 * np.sqrt(np.exp(x[0]) + np.exp(-x[0]) + np.exp(x[1]) + np.exp(-x[1])) - x[0] ** 2 - 0.5 * x[0]
f = [f1, f2, f3]
guesses = [[-2, 2], [-3, 3], [0, 0]]
accuracy = 1000
x_dif = 10
y_dif = 10
for i in range(len(f)):
    points, num_epoch = bfgs(f[i], guesses[i])
    print("For " + str(i + 1) + " function it took " + str(num_epoch) + " iterations!")
    point = points[num_epoch - 1]
    print("Point: " + str(point))
    # paint_contour(point[0] - x_dif, point[0] + x_dif,
    #               point[1] - y_dif, point[1] + y_dif,
    #               accuracy, points, f[i])
    # plt.show()

