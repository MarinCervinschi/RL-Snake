import numpy as np
import random
from config import LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY, MIN_EPSILON
from game.entities import Action, State


class QLearningAgent:
    def __init__(self):
        # 11 boolean sensors -> 2^11 = 2048 states
        self.state_size = 2048
        self.action_size = len(Action)  # Dynamically get size from Enum

        # Row = State Index, Column = Action Index
        self.q_table = np.zeros((self.state_size, self.action_size))

        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON

    # --- REMOVED: get_state_index() is no longer needed here ---

    def get_action(self, state: State) -> Action:
        """
        Decides the next action.
        Input is now the State Object, not a raw vector.
        """
        # 1. Ask the State object for its unique ID
        state_idx = state.to_index()

        # 2. Exploration (Random)
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(Action))

        # 3. Exploitation (Best Q-Value)
        action_idx = np.argmax(self.q_table[state_idx])
        return Action(action_idx)  # Convert int back to Enum

    def train(
        self, state: State, action: Action, reward: float, next_state: State, done: bool
    ):
        """
        Updates Q-Table.
        Inputs are high-level Objects (State, Action).
        """
        # 1. Convert Objects to Indices
        state_idx = state.to_index()
        next_state_idx = next_state.to_index()
        action_idx = action.value  # Get the integer value from Enum (0, 1, or 2)

        # 2. Get current Q
        current_q = self.q_table[state_idx][action_idx]

        # 3. Calculate Target Q (Bellman Equation)
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_idx])
            target_q = reward + (self.gamma * next_max_q)

        # 4. Update Table
        self.q_table[state_idx][action_idx] = current_q + self.lr * (
            target_q - current_q
        )

        # 5. Decay Epsilon
        if done and self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
