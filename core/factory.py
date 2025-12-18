class AgentFactory:
    """Factory for creating RL agents."""

    @staticmethod
    def create_agent(agent_type: str, grid_size: int):
        """
        Create an agent instance.

        Args:
            agent_type: Type of agent ('tabular', 'dqn', 'ppo')
            grid_size: Size of the game grid

        Returns:
            Agent instance

        Raises:
            ValueError: If agent_type is unknown
        """
        agent_type = agent_type.lower()

        print(f"🤖 Creating agent of type: {agent_type} with grid size: {grid_size}")

        if agent_type == "tabular":
            from agents import QLearningAgent

            return QLearningAgent(grid_size=grid_size)

        elif agent_type == "dqn":
            from agents import DQNAgent

            return DQNAgent(grid_size=grid_size)

        elif agent_type == "ppo":
            from agents import PPOAgent

            return PPOAgent(grid_size=grid_size)

        else:
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Must be one of: 'tabular', 'dqn', 'ppo'"
            )


class RendererFactory:
    """Factory for creating game renderers."""

    @staticmethod
    def create_renderer(ui_type: str, grid_size: int, speed: float = 0.1, **kwargs):
        """
        Create a renderer instance.

        Args:
            ui_type: Type of UI ('pygame', 'terminal')
            grid_size: Size of the game grid
            speed: Rendering speed in seconds per frame
            **kwargs: Additional renderer-specific parameters

        Returns:
            Renderer instance

        Raises:
            ValueError: If ui_type is unknown
        """
        ui_type = ui_type.lower()

        if ui_type == "pygame":
            from ui import PyGameRenderer

            return PyGameRenderer(
                width=grid_size,
                height=grid_size,
                cell_size=kwargs.get("cell_size", 30),
                speed=speed,
            )

        elif ui_type == "terminal":
            from ui import TerminalRenderer

            return TerminalRenderer(width=grid_size, height=grid_size, speed=speed)

        else:
            raise ValueError(
                f"Unknown UI type: {ui_type}. " f"Must be one of: 'pygame', 'terminal'"
            )
