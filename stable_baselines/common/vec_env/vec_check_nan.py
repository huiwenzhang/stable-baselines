import warnings

import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper


class VecCheckNan(VecEnvWrapper):
    """
    NaN and inf checking wrapper for vectorized environment, will raise a warning by default,
    allowing you to know from what the NaN of inf originated from.

    :param venv: (VecEnv) the vectorized environment to wrap
    :param raise_exception: (bool) Whether or not to raise a ValueError, instead of a UserWarning
    :param warn_once: (bool) Whether or not to only warn once.
    :param check_inf: (bool) Whether or not to check for +inf or -inf as well
    """

    def __init__(self, venv, raise_exception=False, warn_once=True, check_inf=True):
        VecEnvWrapper.__init__(self, venv)
        self.raise_exception = raise_exception
        self.warn_once = warn_once
        self.check_inf = check_inf
        self._actions = None
        self._observations = None
        self._user_warned = False

    def step_async(self, actions):
        self._check_val(async_step=True, actions=actions)

        self._actions = actions
        self.venv.step_wait(actions)

    def step_wait(self):
        observations, rewards, news, infos = self.venv.step_wait()

        self._check_val(async_step=False, observations=observations, rewards=rewards, news=news, infos=infos)

        self._observations = observations
        return observations, rewards, news, infos

    def reset(self):
        observations = self.venv.reset()
        self._actions = "reset, no first action"

        self._check_val(async_step=False, observations=observations)

        self._observations = observations
        return observations

    def _check_val(self, *, async_step, **kwargs):
        if self._user_warned and self.warn_once:
            return

        found = []
        for name, val in kwargs:
            has_nan = any(np.isnan(val))
            has_inf = self.check_inf and any(np.isinf(val))
            if has_inf:
                found.append((name, "inf"))
            if has_nan:
                found.append((name, "nan"))

        if found:
            self._user_warned = True
            msg = ""
            for i, (name, type_val) in enumerate(found):
                msg +=  "found {} in {}".format(name, type_val)
                if i != len(found) - 1:
                    msg += ", "

            msg += ". Last given value was: "

            if async_step:
                msg += "action={}".format(self._action)
            else:
                msg += "observations={}".format(self._observations)

            if self.raise_exception:
                raise ValueError(msg)
            else:
                warnings.warn(msg, UserWarning)
