import sys
from pso import PSO
import numpy as np

n_particle = int(sys.argv[1])
iteration = int(sys.argv[2])
time_domain = int(sys.argv[3])

# hyperparameters
acceleration = 9.81
velocity = 3.16

bounds_vel = np.array([-30., 30.])
bounds_acc = np.array([-30., 30.])
bounds = [bounds_vel, bounds_acc]
vel_particles = [np.array([0.001, 0.05]), np.array([0.001, 0.05])]


def create_data(vel, acc, N=1000):  # make fake data
    # time domain = [0, 10] sec
    time = np.linspace(0., time_domain, N)
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


pso = PSO(n_particle, bounds, vel_particles, fit, n_iter=iteration)
pso.fit(False)  # save optimization history

if time_domain == 1:
    name = "pso_"+sys.argv[1]
    pso.save_state(name)

my_dict = {"n_particle": n_particle, "iteration": iteration,
           "number point": 1000 / time_domain, "error": float(pso.global_best_value)}

with open('accelerated_motion_results.csv', 'a') as f:  # You will need 'wb' mode in Python 2.x
    import csv
    w = csv.DictWriter(f, my_dict.keys())
    w.writerow(my_dict)
