from pso import PSO
import numpy as np


# hyperparameters
acceleration = 9.81
velocity = 3.16

bounds_vel = np.array([-30., 30.])
bounds_acc = np.array([-30., 30.])
bounds = [bounds_vel, bounds_acc]
vel_particles = [np.array([0.001, 0.5]), np.array([0.001, 0.5])]


def create_data(vel, acc, N=10000):  # make fake data
    time = np.linspace(0., 10000, N)
    x = vel * time + 0.5 * acc * time ** 2
    return (x, time)


# creating data
data = create_data(velocity, acceleration)


def fit(val):
    x, time = data
    res = np.empty(shape=(val.shape[0], 1))
    for i, v in enumerate(val):
        tmp = x - (v[0] * time + 0.5 * v[1] * time ** 2)
        res[i] = np.linalg.norm(tmp, 2)
    return res


pso = PSO(15, bounds, vel_particles, fit, n_iter=1000)
pso.fit(True)  # save optimization history
pso.plot_history()  # plotting history
# pso.save_gif() # save gif
