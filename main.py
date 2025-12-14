from game.engine import SnakeGameEngine
from agents.q_learning_agent import QLearningAgent
from ui.terminal_view import TerminalRenderer
from config import EPISODES, GRID_SIZE
from utils.metrics import TrainingMetrics


def main():
    game = SnakeGameEngine()
    agent = QLearningAgent()
    view = TerminalRenderer(GRID_SIZE, GRID_SIZE, speed=0.1)

    metrics = TrainingMetrics(save_plot=True)

    print("Starting Training...")

    for episode in range(1, EPISODES + 1):

        state_old = game.reset()
        done = False

        steps_taken = 0
        score = 0

        should_render = episode % 100 == 0

        while not done:
            action = agent.get_action(state_old)

            reward, done, score = game.step(action)
            state_new = game.get_state()

            agent.train(state_old, action, reward, state_new, done)

            state_old = state_new

            steps_taken += 1

            if should_render:
                stats = {"episode": episode}
                view.render(game.snake, game.food, score, stats)

        metrics.record_episode(episode, score, steps_taken)

    print("Training Finished. Final Plot...")
    metrics.plot()
    input("Press Enter to close graphs...")


if __name__ == "__main__":
    main()
