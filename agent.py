import numpy as np

class Agent(object):
    def __init__(self):
        pass

    def getAction(self, state, train=False):
        raise NotImplementedError

    def update(self, d):
        """
            Update knowledge based on observation
            d - tuple of (state, action, reward, next_state)
        """
        raise NotImplementedError