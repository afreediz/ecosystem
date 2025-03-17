# config/settings.py - Global simulation settings
from dataclasses import dataclass

# Display settings
WIDTH, HEIGHT = 800, 600
FPS = 30
BACKGROUND_COLOR = (240, 240, 220)  # Light beige for natural look

# Image paths
@dataclass
class IMAGE_PATHS:
    plant = 'assets/plant.png'
    sheep = 'assets/sheep.png'
    fox = 'assets/fox.png'

class Entities_constraints:
    plants_max:int = 100
    fox_max:int = 20
    sheeps_max:int = 50

# Font settings
FONT_NAME = None  # Uses system default font
FONT_SIZE = 24
FONT_COLOR = (0, 0, 0)  # Black