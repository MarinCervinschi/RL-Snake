import numpy as np
import random
from config import LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY, MIN_EPSILON
from game.entities import Action, State


class QLearningAgent:
    def __init__(self):
        # 11 boolean sensors -> 2^11 = 2048 states
        self.state_size = 2048
        self.action_size = len(Action)

        # Row = State Index, Column = Action Index
        self.q_table = np.zeros((self.state_size, self.action_size))

        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON

    def get_action(self, state: State) -> Action:
        """
        Decides the next action.
        """
        state_idx = state.to_index()

        # Exploration (Random)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(Action))

        # Exploitation (Best Q-Value)
        action_idx = np.argmax(self.q_table[state_idx])
        return Action(action_idx)

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ):
        """
        Updates Q-Table.
        """
        state_idx = state.to_index()
        next_state_idx = next_state.to_index()
        action_idx = action.value

        current_q = self.q_table[state_idx][action_idx]

        # Calculate Target Q (Bellman Equation)
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_idx])
            target_q = reward + (self.gamma * next_max_q)

        # Update Table
        self.q_table[state_idx][action_idx] = current_q + self.lr * (
            target_q - current_q
        )

        # Decay Epsilon
        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
