from pso import PSO
import numpy as np
from scipy.integrate import solve_ivp


# LodkaVolterra paramters and initial conditions.
alpha, beta, gamma, delta = 1., 1., 1., 1.
x0, y0 = 4., 2.
tmax = 10

# PSO bounds (by definition parameters >0)
bounds_alpha = np.array([0., 10.])
bounds_beta = np.array([0., 10.])
bounds_gamma = np.array([0., 10.])
bounds_delta = np.array([0., 10.])
bounds = [bounds_alpha, bounds_beta, bounds_gamma, bounds_delta]

# PSO velocities
vel_alpha = np.array([0.001, 0.5])
vel_beta = np.array([0.001, 0.5])
vel_gamma = np.array([0.001, 0.5])
vel_delta = np.array([0.001, 0.5])
vel_particles = [vel_alpha, vel_beta, vel_gamma, vel_delta]


def lodkavolterra(t, X, alpha, beta, gamma, delta):
    x, y = X
    dotx = x * (alpha - beta * y)
    doty = y * (-delta + gamma * x)
    return dotx, doty


def solve_lodkavolterra(lodkavolterra, tmax, x0, y0, alpha, beta, gamma, delta, n=1000):
    soln = solve_ivp(lodkavolterra, (0, tmax), (x0, y0),
                     args=(alpha, beta, gamma, delta),
                     dense_output=True)
    t = np.linspace(0, tmax, n)
    x, y = soln.sol(t)
    return (x, y, t)


def create_data(tmax, x0, y0, alpha, beta, gamma, delta):
    (x, y, t) = solve_lodkavolterra(
        lodkavolterra, tmax, x0, y0, alpha, beta, gamma, delta)
    return (x, y, t)


# creating data
data = create_data(tmax, x0, y0, alpha, beta, gamma, delta)

# defining the fit


def fit(val):
    (x, y, _) = data
    res = np.empty(shape=(val.shape[0], 1))

    for i, v in enumerate(val):

        # for each particle find a solution
        (x_hat, y_hat, _) = solve_lodkavolterra(
            lodkavolterra, tmax, x0, y0, v[0], v[1], v[2], v[3])

        # calculate error norm
        norm_x = np.linalg.norm(x - x_hat, 2)
        norm_y = np.linalg.norm(y - y_hat, 2)

        # putting the result
        res[i] = norm_x + norm_y

    return res


pso = PSO(20, bounds, vel_particles, fit, n_iter=1500)
pso.fit(True)  # save optimization history
