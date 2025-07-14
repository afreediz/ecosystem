# entities/plants.py - Plant classes

import random
import pygame
from entities.base import Entity
from config.settings import WIDTH, HEIGHT, IMAGE_PATHS
from config.parameters import PLANT_PARAMS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.ecosystem import Ecosystem

class Plant(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, IMAGE_PATHS.plant, size=20)
        self.energy = PLANT_PARAMS.initial_energy
        self.growth_rate = PLANT_PARAMS.growth_rate
        self.max_size = PLANT_PARAMS.max_size
        self.reproduction_threshold = PLANT_PARAMS.reproduction_threshold
        self.name = 'plant'
    
    def update(self, ecosystem:'Ecosystem'):
        super().update()
        
        # Plants gain energy from sunlight
        self.energy += PLANT_PARAMS.energy_gain_rate
        
        # Growth
        if self.size < self.max_size:
            self.size += self.growth_rate
            # Update image size
            self.image = pygame.transform.scale(self.original_image, (int(self.size), int(self.size)))
            self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))
        
        # Reproduction
        if self.energy > self.reproduction_threshold and random.random() < PLANT_PARAMS.reproduction_chance:
            if ecosystem.statistics.plants < ecosystem.constrains.plants_max:
                self.reproduce(ecosystem)
    
    def reproduce(self, ecosystem):
        # Create a new plant nearby
        offset_x = random.randint(-20, 20)
        offset_y = random.randint(-20, 20)
        new_x = max(0, min(WIDTH, self.x + offset_x))
        new_y = max(0, min(HEIGHT, self.y + offset_y))
        
        new_plant = Plant(new_x, new_y)
        ecosystem.entities.append(new_plant)
        self.energy -= PLANT_PARAMS.reproduction_cost