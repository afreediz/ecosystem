# environment/ecosystem.py - Ecosystem class

import random
import pygame
from typing import List, Union
from config.settings import WIDTH, HEIGHT, FONT_NAME, FONT_SIZE, FONT_COLOR, DEBUGGER_WIDTH, smallest_entity_size, Entities_constraints
from config.parameters import ENTITY_IN_PRECEPTION
from entities.plant import Plant
from entities.sheep import Sheep
from entities.fox import Fox
from environment.tile import TileTmxMap
from environment.views import Statistics, Climate, Season

black_area = pygame.Surface((DEBUGGER_WIDTH, HEIGHT))
black_area.fill((0,0,0))
GRAY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
cell_size = min(DEBUGGER_WIDTH, smallest_entity_size)

class Ecosystem:
    def __init__(self):
        self.entities: List[Union[Plant, Sheep, Fox]] = []
        self.statistics = Statistics(day=0, plants=0, sheep=0, foxes=0, climate=Climate.SUNNY, season=Season.SPRING)
        self.constrains = Entities_constraints()
        self.tile_map = TileTmxMap(tmx_file=r'C:\Users\Afree\Desktop\AI\fun\ecosystem\ecosystem\data\tiles\map.tmx')
        self.debug = {}
    
    def check_entity_presense(self, bouding_box:tuple, exclude_id) -> int:
        x, y, width, height = bouding_box

        entities_present = []
        for entity in self.entities:
            if x <= entity.x <= (x + width) and y <= entity.y <= ( y + height) and entity.id != exclude_id:
                entities_present.append(entity)

        sheep = False
        fox = False
        plant = False

        for entity in entities_present:
            if isinstance(entity, Sheep):
                sheep = True
            elif isinstance(entity, Fox):
                fox = True
            elif isinstance(entity, Plant):
                plant = True
            
        # sheep_only=1, sheep_and_fox=2, sheep_and_plant=3, fox_only=4, fox_and_plant=5, plant_only=6
        if sheep and fox:
            return ENTITY_IN_PRECEPTION.sheep_and_fox
        elif sheep and plant:
            return ENTITY_IN_PRECEPTION.sheep_and_plant
        elif fox and plant:
            return ENTITY_IN_PRECEPTION.fox_and_plant
        elif sheep:
            return ENTITY_IN_PRECEPTION.sheep
        elif fox:
            return ENTITY_IN_PRECEPTION.fox
        elif plant:
            return ENTITY_IN_PRECEPTION.plant
        else:
            return ENTITY_IN_PRECEPTION.empty

    def populate(self, num_plants=30, num_herbivores=10, num_carnivores=5):
        # Add initial plants
        for _ in range(num_plants):
            x = random.randint(0, WIDTH-DEBUGGER_WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Plant(x, y))
        
        # Add initial herbivores (sheep)
        for _ in range(num_herbivores):
            x = random.randint(0, WIDTH-DEBUGGER_WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Sheep(x, y))
        
        # Add initial carnivores (foxes)
        for _ in range(num_carnivores):
            x = random.randint(0, WIDTH-DEBUGGER_WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Fox(x, y))

    def update(self):
        self.statistics.day += 1
        
        # Update all entities
        for entity in self.entities:
            if entity.alive:
                entity.update(self)
        
        # Remove dead entities
        self.entities = [entity for entity in self.entities if entity.alive]
        
        alive_ids = [x.id for x in self.entities]
        new_debug = {}
        for key, value in self.debug.items():
            if key in alive_ids:
                new_debug[key] = value
        self.debug = new_debug
        
        # Update statistics
        self.statistics.plants = sum(1 for entity in self.entities if isinstance(entity, Plant))
        self.statistics.sheep = sum(1 for entity in self.entities if isinstance(entity, Sheep))
        self.statistics.foxes = sum(1 for entity in self.entities if isinstance(entity, Fox))
        
        # Add some random plants occasionally
        if self.statistics.day % 5 == 0 and self.statistics.plants < 50:
            x = random.randint(0, WIDTH-DEBUGGER_WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Plant(x, y))

        if self.statistics.day % 90 == 0:
            self.change_season()

    def change_season(self):
        current_day = self.statistics.day
        new_season = None

        if current_day == 90:
            new_season = Season.SPRING
            self.constrains.plants_max = 150
        elif current_day == 180:
            new_season = Season.SUMMER
            self.constrains.plants_max = 100
        elif current_day == 270:
            new_season = Season.WINTER
            self.constrains.plants_max = 50
        else:
            self.statistics.day = 0
            new_season = Season.AUTUMN
            self.constrains.plants_max = 80

        if self.statistics.season != new_season:
            self.statistics.season = new_season
            pass # apply seasonal effects
    
    def draw(self, screen):
        # Draw background
        self.tile_map.render_map(screen)
        
        # Draw all entities
        for entity in self.entities:
            entity.draw(screen)
        
        # Draw statistics
        font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        stats_text = f"Day: {self.statistics.day} | Plants: {self.statistics.plants} | Sheep: {self.statistics.sheep} | Foxes: {self.statistics.foxes} | Season : {self.statistics.season.value}"
        text_surface = font.render(stats_text, True, FONT_COLOR)
        screen.blit(text_surface, (10, 10))

        screen.blit(black_area, (WIDTH-DEBUGGER_WIDTH, 0))
        
        y_offset = 10
        for i, (key, value) in enumerate(self.debug.items()):
            perception = value
            # Create a surface to draw perception on
            cell_size = 10  # Size of each block
            surface_width = perception.shape[1] * cell_size
            surface_height = perception.shape[0] * cell_size

            perception_surface = pygame.Surface((surface_width, surface_height))
            perception_surface.fill((0, 0, 0))  # Fill background black

            # Draw perception matrix
            for y in range(perception.shape[0]):
                for x in range(perception.shape[1]):
                    val = perception[y, x]
                    color = GRAY
                    if val == 'prey':
                        color = GREEN
                    elif val == 'predator':
                        color = RED
                    rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
                    pygame.draw.rect(perception_surface, color, rect)

            screen.blit(perception_surface, (WIDTH-DEBUGGER_WIDTH + 10, y_offset))
            y_offset = surface_height + y_offset + 10