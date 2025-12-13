import time
from game.engine import SnakeGameEngine
from agents.q_learning_agent import QLearningAgent
from ui.terminal_view import TerminalRenderer
from config import EPISODES, GRID_SIZE


def main():
    # --- 1. Initialization ---
    game = SnakeGameEngine()
    agent = QLearningAgent()
    view = TerminalRenderer(GRID_SIZE, GRID_SIZE, speed=0.1)

    # Tracking metrics
    scores = []
    total_score = 0
    record = 0

    print("Starting Training...")
    time.sleep(1)

    # --- 2. Training Loop ---
    for episode in range(1, EPISODES + 1):

        # Reset Environment
        state_old = game.reset()
        done = False
        score = 0

        # Determine if we should watch this game (Render every 50th game)
        should_render = (episode % 50 == 0) or (episode == EPISODES)

        while not done:
            # A. Agent chooses action
            action = agent.get_action(state_old)

            # B. Environment executes step
            reward, done, score = game.step(action)
            state_new = game.get_state()

            # C. Train the Brain
            agent.train(state_old, action, reward, state_new, done)

            # D. Update State
            state_old = state_new

            # E. Render (Optional)
            if should_render:
                stats = {"episode": episode, "high_score": record}
                view.render(game.snake, game.food, score, stats)

        # --- End of Episode Metrics ---
        if score > record:
            record = score

        total_score += score
        mean_score = total_score / episode

        if not should_render:
            print(
                f"Episode {episode} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon:.2f}"
            )

    print("\n--- Training Finished ---")
    print(f"Final High Score: {record}")


if __name__ == "__main__":
    main()
