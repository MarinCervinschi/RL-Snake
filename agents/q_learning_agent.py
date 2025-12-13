from core.interfaces import IAgent

class QLearningAgent(IAgent):
    def __init__(self):
        # Initialize Q-table and parameters
        pass

    def get_action(self, state):
        # Return action based on epsilon-greedy policy
        pass

    def train(self, state, action, reward, new_state, done):
        # Update Q-values based on the Q-learning algorithm
        pass