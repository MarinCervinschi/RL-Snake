"""
Play script for visualizing trained agents.

Usage:
    python play.py --agent tabular --grid_size 5
    python play.py --agent dqn --grid_size 10 --episodes 10
    python play.py --agent ppo --grid_size 20 --ui terminal
"""

import time

import click
from tqdm import tqdm

from game.config import GameConfig
from game.engine import SnakeGameEngine
from ui import PyGameRenderer, TerminalRenderer
from utils.agent_utils import AgentType, load_agent


@click.command()
@click.option(
    "--agent",
    type=click.Choice(["tabular", "dqn", "ppo"], case_sensitive=False),
    required=True,
    help="Agent type to play",
)
@click.option(
    "--grid_size",
    type=int,
    default=None,
    help="Grid size (default: 5 for tabular, 10 for dqn, 20 for ppo)",
)
@click.option(
    "--episodes",
    type=int,
    default=5,
    help="Number of episodes to play (default: 5)",
)
@click.option(
    "--ui",
    type=click.Choice(["pygame", "terminal"], case_sensitive=False),
    default="pygame",
    help="UI renderer type (default: pygame)",
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
    help="Custom model path (optional)",
)
def play(
    agent: AgentType,
    grid_size: int | None,
    episodes: int,
    ui: str,
    speed: float,
    model_path: str | None,
):
    """Watch a trained agent play Snake."""

    # Use default grid sizes if not specified
    if grid_size is None:
        grid_size = DEFAULT_GRID_SIZES[agent]

    print(f"🎮 Loading {agent.upper()} agent for {grid_size}×{grid_size} grid...")

    # Load trained agent
    try:
        agent_obj = load_agent(agent, grid_size, model_path)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\n💡 Train the agent first using the notebook:")
        print(f"   notebooks/train_{agent}.ipynb")
        return

    # Create game environment
    config = GameConfig(grid_size=grid_size)
    game = SnakeGameEngine(config)

    # Create renderer
    if ui == "pygame":
        renderer = PyGameRenderer(
            width=grid_size, height=grid_size, cell_size=30, speed=speed
        )
    else:
        renderer = TerminalRenderer(width=grid_size, height=grid_size, speed=speed)

    # Play episodes
    scores = []
    steps_list = []

    try:
        print(f"🎬 Playing {episodes} episodes...\n")

        for episode in tqdm(range(1, episodes + 1), desc="Playing"):
            state = game.reset()
            done = False
            steps = 0

            while not done:
                # Agent selects action
                action = agent_obj.get_action(state)

                # Execute action
                _, done, score = game.step(action)
                next_state = game.get_state()

                state = next_state
                steps += 1

                # Render
                stats = {
                    "episode": episode,
                    "record": max(scores) if scores else 0,
                }
                renderer.render(game.snake, game.food, score, stats)

                time.sleep(speed)

            scores.append(score)
            steps_list.append(steps)

            # Pause between episodes
            if episode < episodes:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Playback interrupted by user")

    finally:
        if ui == "pygame":
            renderer.close()  # type: ignore

    # Print summary
    if scores:
        print("\n" + "=" * 60)
        print("📊 Performance Summary")
        print("=" * 60)
        print(f"Episodes Played:  {len(scores)}")
        print(f"Average Score:    {sum(scores)/len(scores):.2f} apples")
        print(f"Best Score:       {max(scores)} apples")
        print(f"Worst Score:      {min(scores)} apples")
        print(f"Average Steps:    {sum(steps_list)/len(steps_list):.1f}")

        # Efficiency metric
        efficient_episodes = [
            steps_list[i] / scores[i] for i in range(len(scores)) if scores[i] > 0
        ]
        if efficient_episodes:
            print(
                f"Avg Efficiency:   {sum(efficient_episodes)/len(efficient_episodes):.1f} steps/apple"
            )

        print("=" * 60 + "\n")


# Default grid sizes for each agent type
DEFAULT_GRID_SIZES = {
    "tabular": 5,
    "dqn": 10,
    "ppo": 20,
}


if __name__ == "__main__":
    play()
