from game.engine import SnakeGameEngine
from agents.q_learning_agent import QLearningAgent
from ui.terminal_view import TerminalRenderer
from config import EPISODES


def main():
    # Dependency Injection
    game = SnakeGameEngine()
    agent = QLearningAgent()  # Implements IAgent
    view = TerminalRenderer()  # Implements IRenderer

    for episode in range(EPISODES):
        state = game.reset()
        done = False

        while not done:
            # 1. Agent decides
            action = agent.get_action(state)

            # 2. Game updates
            reward, done, score = game.step(action)
            new_state = game.get_state()

            # 3. Agent learns
            agent.train(state, action, reward, new_state, done)

            # 4. View updates (Optional: only every X frames to speed up training)
            view.render(game.get_grid(), score)

            state = new_state


if __name__ == "__main__":
    main()
