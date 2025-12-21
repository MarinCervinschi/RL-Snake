import time

import click
from tqdm import tqdm

from game.config import GameConfig
from game.engine import SnakeGameEngine
from ui import PyGameRenderer
from utils.dqn_play_agent import load_dqn_agent
from utils.metrics import TrainingMetrics


@click.command()
@click.option(
    "--episodes",
    type=int,
    default=5,
    help="Number of episodes to play (default: 5)",
)
@click.option(
    "--speed",
    type=float,
    default=0.1,
    help="Rendering speed in seconds per frame (default: 0.1)",
)
@click.option(
    "--model_path",
    type=str,
    default=None,
    help="Custom model path (default: models/dqn_cnn.pkl)",
)
def play(
    episodes: int,
    speed: float,
    model_path: str | None,
):
    """Watch a trained agent play Snake."""

    try:
        agent_obj = load_dqn_agent(model_path=model_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Train the agent first using the notebook:")
        print(f"   dqn_agent.ipynb")
        return

    config = GameConfig()
    game = SnakeGameEngine(config)
    metrics = TrainingMetrics()
    renderer = PyGameRenderer(cell_size=30, speed=speed)

    try:
        for episode in tqdm(range(1, episodes + 1), desc="Playing"):
            state = game.reset()
            done = False

            while not done:
                action = agent_obj.get_action(state)

                _, done, score = game.step(action)
                next_state = game.get_state()

                state = next_state

                renderer.render(game.snake, game.food, score, metrics.record)

                time.sleep(speed)

            metrics.record_episode(episode, score)

            if episode < episodes:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Playback interrupted by user")

    finally:
        renderer.close()

    metrics.print_summary(play=True)


if __name__ == "__main__":
    play()
