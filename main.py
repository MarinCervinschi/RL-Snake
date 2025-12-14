import click

from config import EPISODES, RENDER_INTERVAL
from core.factory import Factory
from game.engine import SnakeGameEngine
from utils.metrics import TrainingMetrics


@click.command()
@click.option(
    "--ui",
    type=click.Choice(["pygame", "terminal"], case_sensitive=False),
    default="pygame",
    help="Choose the UI renderer (default: pygame)",
)
@click.option(
    "--agent",
    type=click.Choice(["q_learning"], case_sensitive=False),
    default="q_learning",
    help="Choose the agent type (default: q_learning)",
)
@click.option(
    "--show-plots/--no-show-plots",
    default=False,
    help="Whether to show training metric plots (default: no-show-plots)",
)
def main(ui: str, agent_type: str, show_plots: bool):
    """Train a Q-Learning agent to play Snake using Reinforcement Learning."""

    game = SnakeGameEngine()
    agent = Factory.create_agent(agent_type)
    view = Factory.create_renderer(ui)

    metrics = TrainingMetrics()
    record_score = 0

    print(f"Starting Training for {EPISODES} episodes...")
    print("Press Ctrl+C to stop early\n")

    try:
        for episode in range(1, EPISODES + 1):
            state_old = game.reset()
            done = False

            steps_taken = 0
            score = 0

            should_render = episode % RENDER_INTERVAL == 0

            while not done:
                action = agent.get_action(state_old)

                reward, done, score = game.step(action)
                state_new = game.get_state()

                agent.train(state_old, action, reward, state_new, done)

                state_old = state_new

                steps_taken += 1

                if should_render:
                    stats = {"episode": episode, "record": record_score}
                    view.render(game.snake, game.food, score, stats)

            if score > record_score:
                record_score = score

            metrics.record_episode(episode, score, steps_taken)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    finally:
        if ui == "pygame":
            view.close()

        print("\n✅ Training Finished.")
        if show_plots:
            print("Displaying training metrics...")
            metrics.plot()
        input("Press Enter to close graphs...")


if __name__ == "__main__":
    main()
