from pso import PSO
import numpy as np


def poisson_solution(x):
    return -np.sin(np.pi * x)


def discretize(omega, n):
    # choosing the step size in the interval omega
    h = (omega[1] - omega[0]) / (n - 1)

    # discretization points in the interval omega
    x_i = np.array([i * h for i in range(0, n)])
    return x_i


def finDif(omega, f, n, bc):
    """
    Function returning finite different matrix for fourth order difference
    approximation of second order derivative.


    info: Given a one dimensional PDE of the form:

                            -u''(x) = f(x) , insiede boundary
                            u(x) = 0 , in boundary

          The function returns a numpy 2d array A and
          a numpy 1d array b such that:

                              A * u = b

          where u is the solution vector to the PDE
          problem.

    additional: The function uses linear discretization points
                in the domain of the PDE.

    example:    omega = [0,pi]         # domain interval
                f = lambda x : sin(x)
                n=400                  # discretization points
                bc = [0,0]             # boundary conditions

                A, b = finDif(omega, f, n, bc)  # apply function
                u = numpy.linalg.solve(A, b)    # solving linear system
                x = numpy.linspace(omega[0], omega[1], num = numpy.shape(u)[
                                   0]) # discretization points
                plot(x, u)  # plotting the solution

    """
    # choosing the step size in the interval omega
    h = (omega[1] - omega[0]) / (n - 1)

    x_i = discretize(omega, n)

    # calculating the function at discretization points
    b = np.array(f(x_i))

    b[0], b[n - 1] = bc                         # setting boundary conditions
    A = np.diag([-1 for _ in range(0, n - 2)], -2)  \
        + np.diag([16 for _ in range(0, n - 1)], -1) \
        + np.diag([-30 for _ in range(0, n)])         \
        + np.diag([16 for _ in range(0, n - 1)], 1)    \
        + np.diag([-1 for _ in range(0, n - 2)], 2)     # creating the A matrix
    return - A / (12 * (h ** 2)), b


def solve_poisson(omega, force, bc, n=500):
    A, b = finDif(omega, force, n, bc)
    f = np.linalg.solve(A, b)
    return f


def create_poisson_data(omega, n_data=200):
    x = discretize(omega, n_data)
    u = poisson_solution(x)
    data = (x, u)
    return data


# Simulation
omega = [0, 1]
bc = [0, 0]
data = create_poisson_data(omega)


def force(x, val):
    return np.sin(val * x)


def fit(val):
    _, u = data
    res = np.empty(shape=(val.shape[0], 1))
    for i, v in enumerate(val):
        # passing value to the force term
        def f(x):
            return force(x, v)
        # solve poisson equation
        u_hat = solve_poisson(omega, f, bc, u.size)
        # find residual
        res[i] = np.linalg.norm(u_hat - u, 2)
    return res


# setting the bounds
bounds = [np.array([0, np.pi]), np.array([0, np.pi])]
vel_particles = [np.array([0.0001, 0.05]), np.array([0.0001, 0.05])]
pso = PSO(25, bounds, vel_particles, fit, n_iter=500)
pso.fit(True)  # save optimization history
pso.plot_history()  # plotting history
# pso.save_gif() # save gif
