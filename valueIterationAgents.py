# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    actionValues = [self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)]
                    newValues[state] = max(actionValues) if actionValues else 0
            self.values = newValues
        # Write value iteration code here

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        qValue = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + self.discount * self.values[nextState])
        return qValue

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None

        bestAction = None
        bestValue = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            qValue = self.computeQValueFromValues(state, action)
            if qValue > bestValue:
                bestValue = qValue
                bestAction = action
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
