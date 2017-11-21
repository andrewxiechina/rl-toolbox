class EpsilonAnneal(object):
    """
    Annealing epsilon parameter based on ratio of current timestep to total timesteps.
    """

    def __init__(self, initial_epsilon=1.0, final_epsilon=0.1, timesteps=10000, start_timestep=0):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.timesteps = timesteps
        self.start_timestep = start_timestep

    def __call__(self, episode=0, timestep=0):
        if timestep < self.start_timestep:
            return self.initial_epsilon

        elif timestep > self.start_timestep + self.timesteps:
            return self.final_epsilon

        else:
            completed_ratio = (timestep - self.start_timestep) / self.timesteps
            return self.initial_epsilon + completed_ratio * (self.final_epsilon - self.initial_epsilon)


class EpsilonDecay(object):
    """
    Exponentially decaying epsilon parameter based on ratio of
    difference between current and final epsilon to total timesteps.
    """

    def __init__(self, initial_epsilon=1.0, final_epsilon=0.1, timesteps=10000, start_timestep=0, half_lives=10):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.timesteps = timesteps
        self.start_timestep = start_timestep
        self.half_life = self.timesteps / half_lives

    def __call__(self, episode=0, timestep=0):
        if timestep < self.start_timestep:
            return self.initial_epsilon

        elif timestep > self.start_timestep + self.timesteps:
            return self.final_epsilon

        else:
            half_life_ratio = (timestep - self.start_timestep) / self.half_life
            return self.final_epsilon + (2 ** (-half_life_ratio)) * (self.initial_epsilon - self.final_epsilon)