import numpy as np
from calc import gradient, ternary_search_wolfe


def bfgs(f, x0, max_iter=1000, tol=1e-6):
    n = len(x0)
    x = x0
    B = np.eye(n)  # initial approximation of inverse Hessian
    grad = gradient(f, x)
    for ind in range(max_iter):
        p = -np.dot(B, grad)
        alpha = ternary_search_wolfe(f, x, p)
        x_new = x + alpha * p
        s = x_new - x
        grad_new = gradient(f, x_new)
        y = grad_new - grad
        if np.linalg.norm(y) < tol:
            print("It took " + str(ind) + " iterations!")
            break
        rho = 1 / np.dot(y, s)
        B = (np.eye(n) - rho * np.outer(s, y)) @ B @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x, grad = x_new, grad_new
    return x


f1 = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
f2 = lambda x: 0.01 * x[0] ** 2 + x[1] ** 2
f3 = lambda x: (1.5 - x[0] + x[0] * x[1]) ** 2 + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2 + (
        2.625 - x[0] + x[0] * x[1] ** 3) ** 2
f = [f1, f2, f3]
guesses = [[-2, 2], [-3, 3], [0, 0]]

for i in range(len(f)):
    x_opt = bfgs(f[i], guesses[i])
    print(x_opt)
