import mdp, util
from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    def __init__(self, mdp, discount=0.9, iterations=100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Value Iteration 수행
        for i in range(iterations):
            new_values = self.values.copy()
            for state in mdp.getStates():
                if not mdp.isTerminal(state):
                    action_values = []
                    for action in mdp.getPossibleActions(state):
                        q_value = self.computeQValueFromValues(state, action)
                        action_values.append(q_value)
                    new_values[state] = max(action_values)
            self.values = new_values

    def computeQValueFromValues(self, state, action):
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])
        return q_value

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        best_action = None
        best_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getValue(self, state):
        return self.values[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

