from game import Agent
import util

class PacmanAgent(Agent):
    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
    
    def getAction(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

    