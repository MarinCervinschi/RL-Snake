import time
from tqdm import tqdm

import click

from core.factory import AgentFactory, RendererFactory
from game.config import GameConfig
from game.engine import SnakeGameEngine


@click.command()
@click.option(
    "--agent",
    type=click.Choice(["tabular", "dqn", "ppo"], case_sensitive=False),
    default="tabular",
    help="Agent type",
)
@click.option(
    "--grid_size",
    type=int,
    default=None,
    help="Grid size (auto-detected from filename if not specified)",
)
@click.option(
    "--episodes", type=int, default=5, help="Number of episodes to play (default: 5)"
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
    "--greedy/--stochastic",
    default=True,
    help="Use greedy policy (no exploration) vs. stochastic (default: greedy)",
)
def play(
    agent: str,
    grid_size: int,
    episodes: int,
    ui: str,
    speed: float,
    greedy: bool,
):
    """Watch a trained agent play Snake."""

    grid_size = grid_size or GRID_SIZES[agent]

    agent_obj = AgentFactory.create_agent(agent, grid_size)
    agent_obj.load()

    config = GameConfig(grid_size=grid_size)

    game = SnakeGameEngine(config)

    if greedy and hasattr(agent_obj, "epsilon"):
        original_epsilon = agent_obj.epsilon  # type: ignore
        agent_obj.epsilon = 0.0  # No exploration # type: ignore
        print(f"   Set epsilon: {original_epsilon:.4f} → 0.0 (greedy)")

    renderer = RendererFactory.create_renderer(ui, grid_size, speed=speed)

    scores = []
    steps_list = []

    try:
        print(f"🎬 Playing {episodes} episodes...\n")

        for episode in tqdm(range(1, episodes + 1)):

            state = game.reset()
            done = False
            steps = 0

            while not done:
                action = agent_obj.get_action(state)

                _, done, score = game.step(action)
                next_state = game.get_state()

                state = next_state
                steps += 1

                stats = {
                    "episode": episode,
                    "record": max(scores) if scores else 0,
                }
                renderer.render(game.snake, game.food, score, stats)

                time.sleep(speed)

            scores.append(score)
            steps_list.append(steps)

            if episode < episodes:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")

    finally:
        renderer.close()

    if scores:
        print("\n" + "=" * 60)
        print("📊 Evaluation Summary")
        print("=" * 60)
        print(f"Episodes Played:  {len(scores)}")
        print(f"Average Score:    {sum(scores)/len(scores):.2f} apples")
        print(f"Best Score:       {max(scores)} apples")
        print(f"Worst Score:      {min(scores)} apples")
        print(f"Average Steps:    {sum(steps_list)/len(steps_list):.1f}")

        efficient_episodes = [
            steps_list[i] / scores[i] for i in range(len(scores)) if scores[i] > 0
        ]
        if efficient_episodes:
            print(
                f"Avg Efficiency:   {sum(efficient_episodes)/len(efficient_episodes):.1f} steps/apple"
            )

        print("=" * 60 + "\n")


GRID_SIZES = {
    "tabular": 5,
    "dqn": 10,
    "ppo": 20,
}

if __name__ == "__main__":
    play()
