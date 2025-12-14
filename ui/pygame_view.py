import pygame
from core.interfaces import IRenderer
from game.entities import Point
from typing import Optional


class PyGameRenderer(IRenderer):
    def __init__(self, width: int, height: int, cell_size: int = 30, speed: float = 0.1):
        """
        Initialize PyGame renderer.
        
        Args:
            width: Grid width (number of cells)
            height: Grid height (number of cells)
            cell_size: Size of each cell in pixels
            speed: Delay between frames in seconds
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.speed = speed
        
        # Calculate window dimensions
        self.window_width = width * cell_size
        self.window_height = height * cell_size + 60  # Extra space for score
        
        # Initialize PyGame
        pygame.init()
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('RL Snake - Q-Learning')
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Colors (RGB)
        self.colors = {
            'BACKGROUND': (20, 20, 40),
            'GRID': (40, 40, 60),
            'SNAKE_HEAD': (0, 255, 100),
            'SNAKE_BODY': (0, 200, 80),
            'FOOD': (255, 50, 50),
            'TEXT': (255, 255, 255)
        }
        
        # Font for text
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def render(
        self, snake: list, food: Optional[Point], score: int, record: dict
    ) -> None:
        """
        Renders the current game state using PyGame.
        
        Args:
            snake: List of Point objects representing the snake
            food: The location of the food
            score: Current game score
            record: Dictionary containing episode info and other stats
        """
        # Handle PyGame events (to prevent window from freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        
        # Fill background
        self.display.fill(self.colors['BACKGROUND'])
        
        # Draw grid
        self._draw_grid()
        
        # Draw food
        if food:
            self._draw_cell(food.x, food.y, self.colors['FOOD'])
        
        # Draw snake
        for i, segment in enumerate(snake):
            if i == 0:
                # Head - brighter and slightly larger
                self._draw_cell(segment.x, segment.y, self.colors['SNAKE_HEAD'], border=2)
            else:
                # Body
                self._draw_cell(segment.x, segment.y, self.colors['SNAKE_BODY'])
        
        # Draw UI (score, episode)
        self._draw_ui(score, record)
        
        # Update display
        pygame.display.flip()
        
        # Control frame rate
        self.clock.tick(int(1 / self.speed))
    
    def _draw_grid(self):
        """Draw grid lines for better visibility."""
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(
                self.display, 
                self.colors['GRID'], 
                (x, 0), 
                (x, self.window_height - 60)
            )
        for y in range(0, self.window_height - 60, self.cell_size):
            pygame.draw.line(
                self.display, 
                self.colors['GRID'], 
                (0, y), 
                (self.window_width, y)
            )
    
    def _draw_cell(self, x: int, y: int, color: tuple, border: int = 0):
        """
        Draw a single cell on the grid.
        
        Args:
            x: Grid x coordinate
            y: Grid y coordinate
            color: RGB color tuple
            border: Border size in pixels (0 for no border)
        """
        rect = pygame.Rect(
            x * self.cell_size,
            y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        
        if border > 0:
            # Draw with border (for head)
            pygame.draw.rect(self.display, color, rect)
            inner_rect = rect.inflate(-border * 2, -border * 2)
            pygame.draw.rect(self.display, self.colors['BACKGROUND'], inner_rect)
            pygame.draw.rect(self.display, color, inner_rect.inflate(-4, -4))
        else:
            # Draw solid cell
            pygame.draw.rect(self.display, color, rect)
            # Add slight border for better visibility
            pygame.draw.rect(self.display, self.colors['BACKGROUND'], rect, 1)
    
    def _draw_ui(self, score: int, record: dict):
        """Draw score and episode information."""
        ui_y = self.window_height - 55
        
        # Score
        score_text = self.font.render(f'Score: {score}', True, self.colors['TEXT'])
        self.display.blit(score_text, (10, ui_y))
        
        # Episode
        episode = record.get('episode', 0)
        episode_text = self.small_font.render(f'Episode: {episode}', True, self.colors['TEXT'])
        self.display.blit(episode_text, (10, ui_y + 30))
        
        # Record (if available)
        if 'record' in record:
            record_text = self.small_font.render(
                f'Record: {record["record"]}', 
                True, 
                self.colors['TEXT']
            )
            self.display.blit(record_text, (self.window_width - 150, ui_y + 30))
    
    def close(self):
        """Clean up PyGame resources."""
        pygame.quit()
    
    def __del__(self):
        """Ensure PyGame is properly closed."""
        try:
            pygame.quit()
        except:
            pass