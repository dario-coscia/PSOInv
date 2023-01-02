'''Simple testing for pso'''
from pso import PSO
import numpy as np
from numpy.testing import assert_allclose

x_bounds = np.array([-10, 10])
y_bounds = np.array([-10, 10])
bounds = [x_bounds, y_bounds]
vel = [np.array([0.0001, 0.5]), np.array([0.0001, 0.5])]


def ackley(x):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[:, 0]**2 + x[:, 1]**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x[:, 0]) + np.cos(2 * np.pi * x[:, 1]))) + np.e + 20


def sphere(x):
    return x[:, 0]**2 + x[:, 1]**2


def rastrigin(point):
    x, y = point[:, 0], point[:, 1]
    res = 20 + (x**2 - 10*np.cos(2*np.pi*x)) + (y**2 - 10*np.cos(2*np.pi*y))
    return res


def test_pso_constructor():
    pso = PSO(100, bounds, vel, rastrigin, n_iter=1000)


def test_pso_fit():

    pso = PSO(100, bounds, vel, rastrigin, n_iter=1000)
    pso.fit(False)
    val = pso._global_best_val
    assert_allclose(val, np.zeros_like(val), rtol=0., atol=1e-5)

    pso = PSO(100, bounds, vel, sphere, n_iter=1000)
    pso.fit(False)
    val = pso._global_best_val
    assert_allclose(val, np.zeros_like(val), rtol=0., atol=1e-5)

    pso = PSO(100, bounds, vel, ackley, n_iter=1000)
    pso.fit(False)
    val = pso._global_best_val
    assert_allclose(val, np.zeros_like(val), rtol=0., atol=1e-5)
