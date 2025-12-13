from core.interfaces import IRenderer
import os

class TerminalRenderer(IRenderer):
    def render(self, grid_matrix, score):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Score: {score}")
        # Logic to print the matrix as string
        # . . .
        # . O .
        # . * .