"""
Utility to load trained agents for visualization.

This module provides a simple way to load agents without needing
the full factory pattern. Agents are primarily trained in notebooks.
"""

import pickle
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from game.entities import Action, State


AgentType = Literal["tabular", "dqn", "ppo"]


def load_agent(agent_type: AgentType, grid_size: int, model_path: str | None = None):
    """
    Load a trained agent for visualization.

    Args:
        agent_type: Type of agent ("tabular", "dqn", or "ppo")
        grid_size: Grid size the agent was trained on
        model_path: Optional custom model path (uses default if None)

    Returns:
        Loaded agent ready for inference

    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If agent_type is invalid
    """
    if agent_type == "tabular":
        from utils.agent_loaders import load_tabular_agent

        return load_tabular_agent(grid_size, model_path)

    elif agent_type == "dqn":
        from utils.agent_loaders import load_dqn_agent

        return load_dqn_agent(grid_size, model_path)

    elif agent_type == "ppo":
        from utils.agent_loaders import load_ppo_agent

        return load_ppo_agent(grid_size, model_path)

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. Must be 'tabular', 'dqn', or 'ppo'"
        )