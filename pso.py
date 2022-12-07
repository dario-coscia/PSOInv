import sys
import matplotlib.pyplot as plt
import numpy as np
from types import FunctionType
from copy import deepcopy


class PSO(object):
    """Particle Swarm Optimization base class."""

    def __init__(self, swarm_size, boundaries, velocities,
                 fit, velocity_rule=None, n_iter=1000):
        """PSO constructor method

        :param swarm_size: number of particles
        :type swarm_size: int
        :param boundaries: boundaries of positions for optimization
        :type boundaries: list(np.array)
        :param velocities: boundaries of velocity for optimization
        :type velocities: list(np.array)
        :param fit: optimization function
        :type fit: FunctionType
        :param velocity_rule: update velocity rule, defaults to None.
            if `dict` standard velocity update rule is used. The
            dictionary must contain 'social', 'cognitive' and 'inertia'
            components as keys and their value as floating point as
            items. If `None` is passed, default update velocity rule
            is used with 'social' = 1.49445, 'cognitive' = 1.49445,
            and 'inertia': 0.8.
        :type velocity_rule: NoneType, Dict, optional
        :param n_iter: number of iterations, defaults to 1000
        :type n_iter: int, optional
        """

        self._history = None
        self._global_best = None
        self._global_best_val = None

        if isinstance(swarm_size, int):
            self._swarm_pts = swarm_size
        else:
            raise TypeError('expected single integer.')

        if isinstance(fit, FunctionType):
            self._fit = fit
        else:
            raise TypeError('expected python numpy function.')

        if isinstance(n_iter, int):
            self._iter = n_iter
        else:
            raise TypeError('expected single integer.')

        # checks on boundaries and velocities
        if not all(b.size == boundaries[-1].size for b in boundaries):
            raise ValueError(
                'dimensions of boundaries must have same length.')
        if not all(b.size == velocities[-1].size for b in velocities):
            raise ValueError(
                'dimensions of velocities must have same length.')

        if len(boundaries) != len(velocities):
            raise ValueError(
                'length of boundaries and velocity must the same.')

        for i, (bd, v) in enumerate(zip(boundaries, velocities)):

            try:
                if isinstance(bd, np.ndarray) and isinstance(v, np.ndarray):
                    assert bd.size == v.size  # check for dims compatibility
                else:
                    raise TypeError('boundaries and velocities must be tuple '
                                    'or list of numpy.ndarray.')
            except AssertionError:
                raise ValueError('incompatible size for boundaries {bd.size } '
                                 f'and for velocities {v.size} '
                                 f'in dimension {i}.')
        self._bd = boundaries
        self._range_v = velocities
        self._dim = len(velocities)  # dimension of the problem

        # lower and upper bounds boundaries
        self._lower_bounds = []
        self._upper_bounds = []
        for bounds in boundaries:
            self._lower_bounds.append(bounds[0])
            self._upper_bounds.append(bounds[1])

        # lower and upper bounds velocities
        self._lower_bounds_v = []
        self._upper_bounds_v = []
        for bounds in velocities:
            self._lower_bounds_v.append(bounds[0])
            self._upper_bounds_v.append(bounds[1])

        if isinstance(velocity_rule, dict):
            my_keys = ['social', 'cognitive', 'inertia']
            vel_keys = list(velocity_rule.keys())
            present_key = all(key in vel_keys for key in my_keys)
            if not present_key:
                raise ValueError('invalid keys. Please insert `social`, '
                                 '`cognitive` and `inertia` keys.')

            self._parameters_vel = velocity_rule
            self._update_vel = self._standard_velocity_update

        elif velocity_rule is None:
            velocity_rule = {'social': 1.49445,
                             'cognitive': 1.49445,
                             'inertia': 0.8,
                             }
            self._parameters_vel = velocity_rule
            self._update_vel = self._standard_velocity_update

        else:
            raise ValueError(
                'expected python dict or None type for default.')

    def fit(self, verbose=False):
        """Fit method for PSO class, performing PSO optimization.

        :param verbose: shows optimization history on terminal,
             defaults to False
        :type verbose: bool, optional
        """
        self._train(verbose)

    def _standard_velocity_update(self, position, velocity,
                                  global_best, local_best):
        """update velocity method

        :param position: current position
        :type position: np.array
        :param velocity: current velocity
        :type velocity: np.array
        :param global_best: global best between particles
        :type global_best: np.array
        :param local_best: local best for each particle
        :type local_best: np.array
        :return: updated velocity
        :rtype: np.array
        """

        # extracting parameters
        c_cog = self._parameters_vel['cognitive']
        c_soc = self._parameters_vel['social']
        w = self._parameters_vel['inertia']

        # random vectors
        r1 = np.random.random((self._swarm_pts, self._dim))
        r2 = np.random.random((self._swarm_pts, self._dim))

        # calculate three velocity components
        social_component = c_soc * r1 * (global_best - position)
        cognitive_component = c_cog * r2 * (local_best - position)
        inertia = w * velocity

        # update velocity
        velocity = inertia + social_component + cognitive_component

        # handle velocity using damping method
        velocity = self._handle_vel(velocity)
        return velocity

    def _update_position(self, current_position, current_velocity):
        """update position method

        :param current_position: current position
        :type current_position: np.array
        :param current_velocity: current velocity
        :type current_velocity: np.array
        :return: updated position
        :rtype: np.array
        """

        # new position update
        current_position = current_position + current_velocity

        # handle out particles
        new_position = self._handle_pos(current_position)

        # update velocity for boundary match
        # which particle was outside?
        idx = ~np.isclose(new_position, current_position)

        # change velocity for outside boundaries particles
        current_velocity = self._damping(current_velocity, idx)
        return new_position

    def _damping(self, velocity, idx):
        """performing damping on velocity

        :param velocity: input velocity
        :type velocity: np.array
        :param idx: where to peform damping
        :type idx: np.array, velocity.shape must be equal to idx.shape
        :return: damped velocity
        :rtype: np.array
        """

        velocity[idx] = np.random.random() * velocity[idx]
        return velocity

    def _handle_pos(self, x):
        """method handling out of limit positions

        :param x: input position
        :type x: np.array
        :return: position specified by boundaries limits
        :rtype: np.array
        """

        return np.clip(x, self._lower_bounds, self._upper_bounds, out=None)

    def _handle_vel(self, velocity):
        """method handling out of limit velocities

        :param velocity: input velocity
        :type velocity: np.array
        :return: velocity specified by boundaries limits
        :rtype: np.array
        """

        abs_vel = np.abs(velocity)
        clip_val = np.clip(abs_vel, self._lower_bounds_v, self._upper_bounds_v)
        return clip_val * np.sign(velocity)

    def _sample_positions(self):
        """methods for sampling positions

        :return: position samples
        :rtype: np.array
        """
        size = (self._swarm_pts, self._dim)
        positions = np.random.uniform(low=self._lower_bounds,
                                      high=self._upper_bounds,
                                      size=size)
        return positions

    def _sample_velocities(self):
        """methods for sampling velocities

        :return: velocity samples
        :rtype: np.array
        """

        size = (self._swarm_pts, self._dim)
        velocities = np.random.uniform(low=self._lower_bounds_v,
                                       high=self._upper_bounds_v,
                                       size=size)
        sign_velocity = np.random.choice([-1, 1], size=size)
        return velocities * sign_velocity

    def _evaluate_global_best(self, positions):
        """computing global best between particles

        :param positions: current position
        :type positions: np.array
        :return: updated global best
        :rtype: np.array
        """

        # find fit values
        fit_val = self._fit(positions)

        # select minimum
        min_idx = np.argmin(fit_val)

        # update global best values and positions
        global_best = positions[min_idx]
        global_best_val = fit_val[min_idx]
        return global_best, global_best_val

    def _evaluate_local_best(self, local_best, positions):
        """computing local best for each particle

        :param local_best: old local best
        :type local_best: np.array
        :param positions: current position
        :type positions: np.array
        :return: updated local best
        :rtype: np.array
        """

        # find new local bests
        idx = np.where(self._fit(positions) < self._fit(local_best))[0]

        # update local_best array with the new local bests
        local_best[idx] = positions[idx]
        return local_best

    def _progressbar(self, it, size=60):
        """Simple progress bar

        :param it: _description_
        :type it: _type_
        :param prefix: _description_, defaults to ""
        :type prefix: str, optional
        :param size: _description_, defaults to 60
        :type size: int, optional
        :param out: _description_, defaults to sys.stdout
        :type out: _type_, optional
        :yield: _description_
        :rtype: _type_
        """
        count = len(it)

        def show(j):
            x = int(size * j / count)
            print(f"[{u'█'*x}{('.'*(size-x))}] {j + 1}/{count + 1}",
                  end='\r', file=sys.stdout, flush=True)

        print("optimizing...")
        show(0)
        for i, item in enumerate(it):
            yield item
            show(i + 1)
        print("\n", flush=True, file=sys.stdout)

    def _train(self, verbose=False):
        """training loop for pso optimization

        :param verbose: save particles' histories, defaults to False
        :type verbose: bool, optional
        """

        # initialize positions and velocities
        positions = self._sample_positions()
        velocities = self._sample_velocities()

        # find local best for each particle
        local_best = positions

        # find global best between particles
        global_best, global_best_val = self._evaluate_global_best(positions)

        # save history of the particles' positions in a list
        history = [deepcopy(positions)]

        iterator_range = range(1, self._iter)
        if verbose:
            iterator_range = self._progressbar(iterator_range)

        for _ in iterator_range:

            # update velocities
            velocities = self._update_vel(positions,
                                          velocities,
                                          global_best,
                                          local_best)
            # update positions
            positions = self._update_position(positions, velocities)

            # save in history the positions
            history.append(deepcopy(positions))

            # find local best
            local_best = self._evaluate_local_best(local_best, positions)

            # find global best between particles
            global_best, global_best_val = self._evaluate_global_best(
                local_best)

        self._global_best = global_best
        self._global_best_val = global_best_val
        self._history = history

        if verbose:
            print(f"Optimization Summary")
            print(f"    Global best position: {list(self._global_best)}")
            print(f"    Global best value: {self._global_best_val}")

    def plot_history(self, n_points=80):
        """Plotting history for pso optimization."""

        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        if self._dim != 2:
            raise NotImplementedError

        # plotting the domain
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.25)
        plt.set_cmap('cool')
        x = np.linspace(self._bd[0][0], self._bd[0][1], n_points)
        y = np.linspace(self._bd[1][0], self._bd[1][1], n_points)
        x_sp, y_sp = np.meshgrid(x, y)
        coords = np.array([x_sp.reshape(-1,), y_sp.reshape(-1,)]).T
        fit_val = self._fit(coords)
        fit_val = fit_val.reshape((n_points, n_points))
        ax.contourf(x_sp, y_sp, fit_val)

        # plotting population
        pop = self.history[0]
        scatt = ax.scatter(pop[:, 0], pop[:, 1], color='yellow')

        axslider = fig.add_axes([0.25, 0.1, 0.65, 0.03])

        slider = Slider(ax=axslider, label='iteration', valmin=1,
                        valmax=self._iter, valstep=1)

        def update(val):
            i = int(slider.val)-1
            pop = self.history[i]
            scatt.set_offsets(pop)
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    def save_gif(self, title=None, step=1, fps=20):
        """Save the evolution history as a .gif file.

        :param title: title for the gif, defaults to None
        :type title: string, optional
        :param step: step size in history, defaults to 1
        :type step: int, optional
        :param fps: frame per second in gif, defaults to 20
        :type fps: int, optional
        """

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter

        if self._dim != 2:
            raise NotImplementedError

        def __create_fig(t):
            # plotting the domain
            plt.set_cmap('cool')
            n_points = 80
            x = np.linspace(self._bd[0][0], self._bd[0][1], n_points)
            y = np.linspace(self._bd[1][0], self._bd[1][1], n_points)
            x_sp, y_sp = np.meshgrid(x, y)
            coords = np.array([x_sp.reshape(-1,), y_sp.reshape(-1,)]).T
            fit_val = self._fit(coords)
            fit_val = fit_val.reshape((n_points, n_points))
            ax.contourf(x_sp, y_sp, fit_val)
            # plotting population
            pop = self.history[t]
            ax.scatter(pop[:, 0], pop[:, 1], color='yellow')
            return

        fig, ax = plt.subplots()

        # creating the snaps
        anim_created = FuncAnimation(fig,
                                     __create_fig,
                                     frames=self._iter - 1,
                                     interval=step)

        if title is None:
            title = 'pso_evolution'

        anim_created.save(title + '.gif', dpi=100,
                          writer=PillowWriter(fps=fps))
        plt.close()

    @ property
    def history(self):
        """Property decorator for history

        :return: history of particles
        :rtype: list(np.array)
        """
        return self._history

    @property
    def global_best_position(self):
        return self._global_best

    @property
    def global_best_value(self):
        return self._global_best_val
