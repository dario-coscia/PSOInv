import numpy as np
from math import exp
from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):

    def __init__(self, velocity_rule_dict=None):

        if velocity_rule_dict is None:
            velocity_rule = {'social': 1.49445,
                             'cognitive': 1.49445,
                             'inertia': 0.8,
                             }
            self._velocity_rule_dict = velocity_rule
            self._velocity_rule_dict_original = velocity_rule

        else:
            # check consistency
            self._check_consistency_all(velocity_rule_dict, all)
            self._velocity_rule_dict = velocity_rule_dict
            self._velocity_rule_dict_original = velocity_rule_dict

    @abstractmethod
    def update(self, time):
        pass

    def __call__(self, time):
        self.update(time)

    @ property
    def parameters_dict(self):
        return self._velocity_rule_dict

    @property
    def c_soc(self):
        return self._velocity_rule_dict['social']

    @property
    def c_cog(self):
        return self._velocity_rule_dict['cognitive']

    @property
    def w(self):
        return self._velocity_rule_dict['inertia']

    def _check_consistency(self, velocity_rule_dict, type=all):

        if isinstance(velocity_rule_dict, dict):
            my_keys = ['social', 'cognitive', 'inertia']
            vel_keys = list(velocity_rule_dict.keys())
            present_key = type(key in vel_keys for key in my_keys)
            if not present_key:
                raise ValueError('invalid keys. Please insert `social`, '
                                 '`cognitive` and `inertia` keys.')
        else:
            raise ValueError('expected a dict with keys: social`, '
                             '`cognitive` and `inertia` keys.')


class VelocityExponentialScheduler(Scheduler):

    def __init__(self, vel_initial_params=None, decays_value=None):
        super().__init__(velocity_rule_dict=vel_initial_params)

        if decays_value is None:
            self._decay = self._velocity_rule_dict_original

        # checking keys consistency
        else:
            super()._check_consistency(decays_value, any)
            self._decay = decays_value

    def update(self, time):
        return self._update(time)

    def _update(self, time):
        for key, value in self._decay.items():
            param0 = self._velocity_rule_dict_original[key]
            self._velocity_rule_dict[key] = param0 * exp(-time * value)
