import pygame, csv
from config.settings import TileImages, TILE_SIZE, TILE_MAP_LAYERS

class TileMap():
    def __init__(self, tile_map_layers=TILE_MAP_LAYERS, tile_images=TileImages, tile_size=TILE_SIZE) -> None:
        self.tile_size = tile_size
        self.images_relative_path = 'assets/images/'

        self.tileimages = self.load_images(tile_images)
        # Load multiple layers
        self.tile_map_layers = {}
        for layer_name, layer_file in tile_map_layers.items():
            self.tile_map_layers[layer_name] = self.load_tilemap(layer_file)

    def load_images(self, images_class:dict) -> dict:
        """Load images into a dictionary with keys as tile names."""
        images = {}
        for name, path in images_class.items():  # Get class attributes
            images[name] = pygame.image.load(self.images_relative_path + path)
        return images

    def render(self, screen:pygame.Surface) -> None:
        # Draw each layer in order
        for layer_name, layer_map in self.tile_map_layers.items():
            for y, row in enumerate(layer_map):
                for x, tile in enumerate(row):
                    tile = tile.strip()  # Remove extra spaces
                    if tile != -1:
                        tile_image = self.get_tile_image(layer_name, tile)
                        if tile_image:
                            screen.blit(tile_image, (x * self.tile_size, y * self.tile_size))

    def get_tile_image(self, layer_name, tile_id):
        """Return the appropriate tile image based on layer and tile ID."""
        # Map tile IDs to image keys based on the layer
        # This is a simple example - you might need a more complex mapping
        
        # Example mapping logic
        if layer_name == "base":
            if tile_id == '0':
                return self.tileimages.get('grass', None)
            elif tile_id == '1':
                return self.tileimages.get('water', None)
        elif layer_name == "plants":
            if tile_id == '1':
                return self.tileimages.get('tree', None)
            elif tile_id == '21':
                return self.tileimages.get('rock', None)
            elif tile_id == '2':
                return self.tileimages.get('rock', None)
            elif tile_id == '2':
                return self.tileimages.get('rock', None)
            elif tile_id == '2':
                return self.tileimages.get('rock', None)
            elif tile_id == '2':
                return self.tileimages.get('rock', None)
            elif tile_id == '2':
                return self.tileimages.get('rock', None)
        return None  # Return None if no mapping found

    def load_tilemap(self, filename):
        with open(filename, newline='') as file:
            reader = csv.reader(file)
            return [row for row in reader]