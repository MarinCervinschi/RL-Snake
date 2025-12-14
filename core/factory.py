from agents.q_learning_agent import QLearningAgent
from config import GRID_SIZE, RENDER_SPEED
from core.interfaces import IAgent, IRenderer
from ui.pygame_view import PyGameRenderer
from ui.terminal_view import TerminalRenderer


class Factory:
    @staticmethod
    def create_renderer(ui_type: str) -> IRenderer:
        if ui_type == "pygame":
            print("Using PyGame UI.")
            return PyGameRenderer(
                GRID_SIZE, GRID_SIZE, cell_size=30, speed=RENDER_SPEED
            )

        elif ui_type == "terminal":
            print("Using Terminal UI.")
            return TerminalRenderer(GRID_SIZE, GRID_SIZE, speed=RENDER_SPEED)
        else:
            raise ValueError(f"Unknown UI type: {ui_type}")

    @staticmethod
    def create_agent(agent_type: str) -> IAgent:
        if agent_type == "q_learning":
            print("Using Q-Learning Agent.")
            return QLearningAgent()
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
