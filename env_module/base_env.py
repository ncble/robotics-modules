import gym
from gym.utils import seeding


class BaseEnv(gym.Env):
    metadata = {
        'video.frames_per_second': 50
    }
    def __init__(self, *, srl_model, env_rank, srl_pipe):
        """
        Gym wrapper for robotic environments

        :param srl_model: (str) The SRL_model used
        :param env_rank: (int) the number ID of the environment
        :param srl_pipe: (Queue, [Queue]) contains the input and output of the SRL model
        """
        # the * here, means that the rest of the args need to be called as kwargs.
        # This is done to avoid unwanted situations where we might add a parameter
        #  later and not realise that srl_pipe was not set by an unchanged subclass.
        self.env_rank = env_rank
        self.srl_pipe = srl_pipe
        self.srl_model = srl_model
        self.np_random = None

        # Create numpy random generator
        # This seed can be changed later
        self.seed(0)

    def getSRLState(self, observation):
        """
        get the SRL state for this environement with a given observation
        :param observation: (numpy float) image
        :return: (numpy float)
        """
        if self.srl_model == "ground_truth":
            return self.getGroundTruth()
        else:
            # srl_pipe is a tuple that contains:
            #  Queue: input to the SRL model, sends origin (where does the message comes from, here the rank of the environment)
            #  and observation that needs to be transformed into a state
            #  [Queue]: input for all the envs, sends state associated to the observation
            self.srl_pipe[0].put((self.env_rank, observation))
            return self.srl_pipe[1][self.env_rank].get()

    @staticmethod
    def getGroundTruthDim():
        """
        :return: (int)
        """
        raise NotImplementedError()

    def getGroundTruth(self):
        """
        Get handcrafted (feature engineering) ground truth
        :return: (numpy array)
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """
        Seed random generator
        :param seed: (int)
        :return: ([int])
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :param action: (int or [float])
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the environment
        :return: (numpy tensor) first observation of the env
        """
        raise NotImplementedError()

    def render(self):
        """
        :return: (numpy array)
        """
        raise NotImplementedError()
