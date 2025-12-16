import click
from tqdm import tqdm

from core.factory import AgentFactory, RendererFactory
from game.config import GameConfig
from game.engine import SnakeGameEngine
from utils.metrics import TrainingMetrics


@click.command()
@click.option(
    "--agent",
    type=click.Choice(["tabular", "dqn", "ppo"], case_sensitive=False),
    default="tabular",
    help="Agent type to train",
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
    default=None,
    help="Number of training episodes (default: agent-specific)",
)
@click.option(
    "--render/--no-render",
    default=False,
    help="Render training (default: no-render for speed)",
)
@click.option(
    "--render_every",
    type=int,
    default=100,
    help="Render every N episodes (default: 100)",
)
@click.option(
    "--ui",
    type=click.Choice(["pygame", "terminal"], case_sensitive=False),
    default="pygame",
    help="UI renderer type (if rendering enabled) (default: pygame)",
)
@click.option(
    "--show_plots/--no-show_plots",
    default=False,
    help="Show training plots after training (default: no-show)",
)
@click.option(
    "--save_plots/--no-save_plots",
    default=False,
    help="Save training plots to file (default: no-save)",
)
def train(
    agent: str,
    grid_size: int,
    episodes: int,
    render: bool,
    render_every: int,
    ui: str,
    show_plots: bool,
    save_plots: bool,
):
    """Train a reinforcement learning agent to play Snake."""

    grid_size = grid_size or GRID_SIZES[agent]
    episodes = episodes or EPISODES[agent]

    config = GameConfig(grid_size=grid_size)
    game = SnakeGameEngine(config)

    agent_obj = AgentFactory.create_agent(agent, grid_size)

    renderer = None
    if render:
        renderer = RendererFactory.create_renderer(ui, grid_size)

    metrics = TrainingMetrics(save_dir=f"results/{agent}_{grid_size}x{grid_size}")

    print(f"🚀 Starting training...\n")
    record_score = 0

    try:
        for episode in tqdm(range(1, episodes + 1), desc="Training"):
            state = game.reset()
            done = False
            episode_reward = 0
            steps = 0

            should_render = render and (episode % render_every == 0)

            while not done:
                action = agent_obj.get_action(state)

                reward, done, score = game.step(action)
                next_state = game.get_state()

                agent_obj.train(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                steps += 1

                if should_render and renderer:
                    stats = {
                        "episode": episode,
                        "record": record_score,
                    }
                    renderer.render(game.snake, game.food, score, stats)

            if score > record_score:
                record_score = score

            metrics.record_episode(episode, score, steps, episode_reward)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    finally:
        if renderer:
            renderer.close()

    agent_obj.save()
    metrics.print_summary()

    if show_plots or save_plots:
        metrics.plot(show=show_plots, save=save_plots)
        input("Press Enter to continue...") if show_plots else None

    print("\n✨ All done!\n")


EPISODES = {
    "tabular": 10000,
    "dqn": 3000,
    "ppo": 5000,
}

GRID_SIZES = {
    "tabular": 5,
    "dqn": 10,
    "ppo": 20,
}

if __name__ == "__main__":
    train()
