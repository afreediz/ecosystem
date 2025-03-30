# config/settings.py - Global simulation settings
from dataclasses import dataclass

# Display settings
WIDTH, HEIGHT = 1400, 800
FPS = 30
BACKGROUND_COLOR = (240, 240, 220)  # Light beige for natural look

TILE_SIZE = 16
TILE_MAP_LAYERS = {
    'base':r"C:\Users\Afree\Desktop\AI\fun\ecosystem\ecosystem\data\tiles\base.csv",
    'plants':r"C:\Users\Afree\Desktop\AI\fun\ecosystem\ecosystem\data\tiles\map_plants.csv"
}

# Image paths
@dataclass
class IMAGE_PATHS:
    plant = 'entities/plant.png'
    sheep = 'entities/sheep.png'
    fox = 'entities/fox.png'

TileImages = {
    'grass': 'tiles/Grass_Middle.png',
    'sand': 'tiles/Sand_Middle.png',
    'water': 'tiles/Water_Middle.png',
    'cliff': 'tiles/Cliff_Tile.png',
    'water_sand': 'tiles/Beach_Tile.png',
    'grass_sand': 'tiles/Grass_Sand.png',
    'grass_water': 'tiles/Water_Tile.png',
    'farm_land': 'tiles/FarmLand_Tile.png'
}

DecorImages = {
    'bridge': 'decor/Bridge_Wood.png',
    'oak_tree': 'decor/Oak_Tree.png',
    'decor': 'decor/decorations.png'
}

class Entities_constraints:
    plants_max:int = 100
    fox_max:int = 20
    sheeps_max:int = 50

# Font settings
FONT_NAME = None  # Uses system default font
FONT_SIZE = 24
FONT_COLOR = (0, 0, 0)  # Black