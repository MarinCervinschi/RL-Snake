# Game Settings
GRID_SIZE = 20

# RL Hyperparameters
LEARNING_RATE = 0.1       # Alpha: Common starting point
DISCOUNT_FACTOR = 0.9     # Gamma: We care significantly about future rewards
EPSILON = 1.0             # Start by exploring 100% of the time
EPSILON_DECAY = 0.995     # Reduce exploration slightly every game
MIN_EPSILON = 0.01        # Always explore at least 1% of the time

EPISODES = 1000           # How many games to train

# Rendering Settings
RENDER_SPEED = 0.1        # Seconds between frames when rendering
RENDER_INTERVAL = 100     # Render every N episodes