from pso import PSO
import numpy as np
from scipy.integrate import odeint

saving = True

# For the math see https://scipython.com/blog/the-double-pendulum/
# Script adapted from https://scipython.com/blog/the-double-pendulum/


# Hyperparameters.

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 5.3, 3.8  # to be found via PSO
M1, M2 = 2.1, 1.0

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 10, 0.001
t = np.arange(0, tmax + dt, dt)

# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3 * np.pi / 7, 0, 3 * np.pi / 4, 0])

# The gravitational acceleration (m.s-2).
g = 9.81

# PSO bounds (by definition parameters >0)
bounds_L1 = np.array([1e-4, 30.])
bounds_L2 = np.array([1e-4, 30.])
bounds = [bounds_L1, bounds_L2]

# PSO velocities
vel_L1 = np.array([0.0001, 0.5])
vel_L2 = np.array([0.0001, 0.5])
vel_particles = [vel_L1, vel_L2]


def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c /
                                                     + L2 * z2 ** 2) /
             - (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)
    theta2dot = z2
    z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta2) /
                          + g * np.sin(theta1) * c) /
             + m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)
    return theta1dot, z1dot, theta2dot, z2dot


def solve_double_pendulum(deriv, y0, t, L1, L2, m1, m2):

    y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

    # Unpack z and theta as a function of time
    theta1, theta2 = y[:, 0], y[:, 2]

    # Convert to Cartesian coordinates of the two bob positions.
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return (x1, y1, x2, y2)


def create_data(deriv, y0, t, L1, L2, m1, m2):
    (x1, y1, x2, y2) = solve_double_pendulum(deriv, y0, t, L1, L2, m1, m2)
    return (x1, y1, x2, y2)


# creating data
data = create_data(deriv, y0, t, L1, L2, M1, M2)


def fit(val):
    (x1, y1, x2, y2) = data
    res = np.empty(shape=(val.shape[0], 1))

    for i, v in enumerate(val):

        # for each particle find a solution
        (x1_hat, y1_hat, x2_hat, y2_hat) = solve_double_pendulum(
            deriv, y0, t, v[0], v[1], M1, M2)

        # calculate error norm
        norm_x1 = np.linalg.norm(x1 - x1_hat, 2)
        norm_y1 = np.linalg.norm(y1 - y1_hat, 2)
        norm_x2 = np.linalg.norm(x2 - x2_hat, 2)
        norm_y2 = np.linalg.norm(y2 - y2_hat, 2)

        # putting the result
        res[i] = norm_x1 + norm_y1 + norm_x2 + norm_y2

    return res


pso = PSO(20, bounds, vel_particles, fit, n_iter=100)
pso.fit(verbose=True)

if saving:
    pso.save_state("double_pendulum")
