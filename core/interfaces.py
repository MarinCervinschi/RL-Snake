from abc import ABC, abstractmethod

class IRenderer(ABC):
    @abstractmethod
    def render(self, state_grid, score):
        pass

class IAgent(ABC):
    @abstractmethod
    def get_action(self, state):
        pass
    
    @abstractmethod
    def train(self, state, action, reward, next_state, done):
        pass