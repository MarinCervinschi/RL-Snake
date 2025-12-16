from typing import Optional

import pygame

from core.interfaces import IRenderer
from game.entities import Point


class PyGameRenderer(IRenderer):
    def __init__(
        self, width: int, height: int, cell_size: int = 30, speed: float = 0.1
    ):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.speed = speed

        # Dimensions
        self.grid_pixel_width = width * cell_size
        self.grid_pixel_height = height * cell_size

        # Minimum window width to prevent UI overlap
        MIN_WINDOW_WIDTH = 320
        self.window_width = max(self.grid_pixel_width, MIN_WINDOW_WIDTH)
        self.window_height = self.grid_pixel_height + 60

        # Centering offset
        self.offset_x = (self.window_width - self.grid_pixel_width) // 2

        pygame.init()
        self.display = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("RL Snake - Q-Learning")
        self.clock = pygame.time.Clock()

        self.colors = {
            "BACKGROUND": (20, 20, 40),
            "GRID": (40, 40, 60),
            "BORDER": (100, 100, 120),  # New lighter color for the border
            "SNAKE_HEAD": (0, 255, 100),
            "SNAKE_BODY": (0, 200, 80),
            "FOOD": (255, 50, 50),
            "TEXT": (255, 255, 255),
        }

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def render(
        self, snake: list, food: Optional[Point], score: int, record: dict
    ) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.display.fill(self.colors["BACKGROUND"])

        # Draw grid and border
        self._draw_grid()

        if food:
            self._draw_cell(food.x, food.y, self.colors["FOOD"])

        for i, segment in enumerate(snake):
            if i == 0:
                self._draw_cell(
                    segment.x, segment.y, self.colors["SNAKE_HEAD"], border=2
                )
            else:
                self._draw_cell(segment.x, segment.y, self.colors["SNAKE_BODY"])

        self._draw_ui(score, record)

        pygame.display.flip()
        self.clock.tick(int(1 / self.speed))

    def _draw_grid(self):
        """Draw grid lines and a border around the play area."""
        # 1. Draw internal grid lines
        for x in range(0, self.grid_pixel_width + 1, self.cell_size):
            draw_x = x + self.offset_x
            pygame.draw.line(
                self.display,
                self.colors["GRID"],
                (draw_x, 0),
                (draw_x, self.grid_pixel_height),
            )
        for y in range(0, self.grid_pixel_height + 1, self.cell_size):
            pygame.draw.line(
                self.display,
                self.colors["GRID"],
                (self.offset_x, y),
                (self.offset_x + self.grid_pixel_width, y),
            )

        # 2. Draw the border rectangle
        border_rect = pygame.Rect(
            self.offset_x,  # X position (centered)
            0,  # Y position (top)
            self.grid_pixel_width,  # Width of grid only
            self.grid_pixel_height,  # Height of grid only
        )

        # Draw the border with a thickness of 2 or 3 pixels
        pygame.draw.rect(self.display, self.colors["BORDER"], border_rect, 3)

    def _draw_cell(self, x: int, y: int, color: tuple, border: int = 0):
        # Apply offset_x to cell drawing
        rect = pygame.Rect(
            self.offset_x + (x * self.cell_size),
            y * self.cell_size,
            self.cell_size,
            self.cell_size,
        )

        if border > 0:
            pygame.draw.rect(self.display, color, rect)
            inner_rect = rect.inflate(-border * 2, -border * 2)
            pygame.draw.rect(self.display, self.colors["BACKGROUND"], inner_rect)
            pygame.draw.rect(self.display, color, inner_rect.inflate(-4, -4))
        else:
            pygame.draw.rect(self.display, color, rect)
            pygame.draw.rect(self.display, self.colors["BACKGROUND"], rect, 1)

    def _draw_ui(self, score: int, record: dict):
        ui_y = self.window_height - 55

        score_text = self.font.render(f"Score: {score}", True, self.colors["TEXT"])
        self.display.blit(score_text, (10, ui_y))

        episode = record.get("episode", 0)
        episode_text = self.small_font.render(
            f"Episode: {episode}", True, self.colors["TEXT"]
        )
        self.display.blit(episode_text, (10, ui_y + 30))

        if "record" in record:
            record_text = self.small_font.render(
                f'Record: {record["record"]}', True, self.colors["TEXT"]
            )
            text_rect = record_text.get_rect()
            text_rect.topright = (self.window_width - 10, ui_y + 30)
            self.display.blit(record_text, text_rect)

    def close(self):
        pygame.quit()

    def __del__(self):
        try:
            pygame.quit()
        except:
            pass
