from game import *
from learningAgents import ReinforcementAgent
import random, util, math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # Initialize Q-values
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action) where the max is over legal actions.
          Note that if there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max([self.getQValue(state, action) for action in actions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state. Note that if there are
          no legal actions, which is the case at the terminal state, you should
          return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None

        max_value = self.computeValueFromQValues(state)
        best_actions = [action for action in actions if self.getQValue(state, action) == max_value]
        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state. With probability self.epsilon,
          we should take a random action and take the best policy action otherwise.
          Note that if there are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legalActions = self.getLegalActions(state)
        action = None
        if legalActions:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a state = action => nextState and reward transition.
          You should do your Q-Value update here.

          Good Terminal state = 0.0
          Bad Terminal state = -1.0
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

