from pso import PSO
import numpy as np
from scipy.integrate import solve_ivp


# Lorenz paramters and initial conditions.
sigma, beta, rho = 10, 2.667, 28
u0, v0, w0 = 0, 1, 1.05
tmax = 10

# PSO bounds (by definition parameters >0)
bounds_sigma = np.array([0., 30.])
bounds_beta = np.array([0., 5.])
bounds_rho = np.array([0., 30.])
bounds = [bounds_sigma, bounds_beta, bounds_rho]

# PSO velocities
vel_sigma = np.array([0.001, 0.05])
vel_beta = np.array([0.001, 0.05])
vel_rho = np.array([0.001, 0.05])
vel_particles = [vel_sigma, vel_beta, vel_rho]


def lorenz(t, X, sigma, beta, rho):
    """The Lorenz equations."""
    u, v, w = X
    up = - sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return up, vp, wp


def solve_lorenz(lorenz, tmax, u0, v0, w0, sigma, beta, rho, n=1000):
    """Solving lorenz equation."""
    soln = solve_ivp(lorenz, (0, tmax), (u0, v0, w0), args=(sigma, beta, rho),
                     dense_output=True)
    t = np.linspace(0, tmax, n)
    x, y, z = soln.sol(t)
    return (x, y, z, t)


def create_data(tmax, u0, v0, w0, sigma, beta):
    (x, y, z, t) = solve_lorenz(lorenz, tmax, u0, v0, w0, sigma, beta, rho)
    return (x, y, z, t)


# creating data
data = create_data(tmax, u0, v0, w0, sigma, beta)

# defining the fit


def fit(val):
    (x, y, z, _) = data
    res = np.empty(shape=(val.shape[0], 1))

    for i, v in enumerate(val):

        # for each particle find a solution
        (x_hat, y_hat, z_hat, _) = solve_lorenz(
            lorenz, tmax, u0, v0, w0, v[0], v[1], v[2])

        # calculate error norm
        norm_x = np.linalg.norm(x - x_hat, 2)
        norm_y = np.linalg.norm(y - y_hat, 2)
        norm_z = np.linalg.norm(z - z_hat, 2)

        # putting the result
        res[i] = norm_x + norm_y + norm_z

    return res


pso = PSO(20, bounds, vel_particles, fit, n_iter=500)
pso.fit(True)  # save optimization history
# pso.plot_history()  # plotting history
# pso.save_gif() # save gif
