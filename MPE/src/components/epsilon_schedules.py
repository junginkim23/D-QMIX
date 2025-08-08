import numpy as np


class DecayThenFlatSchedule:

    def __init__(self,
                 start: float,
                 finish: float,
                 time_length: int,
                 decay: str = "exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, t: int):

        if self.decay in ["linear"]:

            return max(self.finish, self.start - self.delta * t)

        elif self.decay in ["exp"]:

            return min(self.start, max(self.finish, np.exp(- t / self.exp_scaling)))

    pass


class DecayThenFlatScheduleForRODE:

    def __init__(self,
                 start: float,
                 finish: float,
                 time_length: int,
                 time_length_exp: int,
                 role_action_spaces_update_start: int,
                 decay: str = "exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay
        self.role_action_spaces_update_start = role_action_spaces_update_start
        self.reset = True
        self.time_length_exp = time_length_exp
        self.start_t = 0

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, t: int):

        if t > self.role_action_spaces_update_start and self.reset:
            self.reset = False
            self.time_length = self.time_length_exp
            self.delta = (self.start - self.finish) / self.time_length
            self.start_t = t

        if self.decay in ["linear"]:

            return max(self.finish, self.start - self.delta * (t-self.start_t))

        elif self.decay in ["exp"]:

            return min(self.start, max(self.finish, np.exp(- t / self.exp_scaling)))

    pass