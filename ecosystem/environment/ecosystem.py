# environment/ecosystem.py - Ecosystem class

import random
import pygame
from typing import List, Union
from config.settings import WIDTH, HEIGHT, FONT_NAME, FONT_SIZE, FONT_COLOR, FRAMES_PER_DAY, DAYS_PER_MONTH, MONTHS_PER_YEAR, Entities_constraints
from entities.plant import Plant
from entities.sheep import Sheep
from entities.fox import Fox
from environment.tile import TileTmxMap
from environment.views import Statistics, Climate, Season

class Ecosystem:
    def __init__(self):
        self.entities: List[Union[Plant, Sheep, Fox]] = []
        self.statistics = Statistics(
            plants=0, sheep=0, foxes=0, 
            climate=Climate.SUNNY, season=Season.SPRING,
            frame=0, day=1, month=1, year=1
            )
        self.constrains = Entities_constraints()
        self.tile_map = TileTmxMap(tmx_file=r'C:\Users\Afree\Desktop\AI\fun\ecosystem\ecosystem\data\tiles\map.tmx')
    
    def populate(self, num_plants=30, num_herbivores=10, num_carnivores=5):
        # Add initial plants
        for _ in range(num_plants):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Plant(x, y))
        
        # Add initial herbivores (sheep)
        for _ in range(num_herbivores):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Sheep(x, y))
        
        # Add initial carnivores (foxes)
        for _ in range(num_carnivores):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Fox(x, y))
    
    def update(self):
        self.statistics.frame += 1
        
        if self.statistics.frame % FRAMES_PER_DAY == 0:
            self.statistics.day += 1
        
        if self.statistics.day % DAYS_PER_MONTH == 0:
            self.statistics.day = 1
            self.statistics.month += 1
            self.change_season()

        if self.statistics.month % MONTHS_PER_YEAR == 0:
            self.statistics.month = 1
            self.statistics.year += 1
            for entity in self.entities:
                if entity.alive:
                    entity.age += 1

        # Update all entities
        for entity in self.entities:
            if entity.alive:
                entity.update(self)
        
        # Remove dead entities
        self.entities = [entity for entity in self.entities if entity.alive]
        
        # Update statistics
        self.statistics.plants = sum(1 for entity in self.entities if isinstance(entity, Plant))
        self.statistics.sheep = sum(1 for entity in self.entities if isinstance(entity, Sheep))
        self.statistics.foxes = sum(1 for entity in self.entities if isinstance(entity, Fox))
        
        # Add some random plants per 5 days
        if self.statistics.day % 5 == 0 and self.statistics.plants < 50:
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Plant(x, y))

    def change_season(self):
        new_season = None
        month = self.statistics.month
        quarter = MONTHS_PER_YEAR / 4
        
        # change season every 3 months
        if month < quarter * 1:
            new_season = Season.SPRING
            self.constrains.plants_max = 150
        elif month < quarter * 2:
            new_season = Season.SUMMER
            self.constrains.plants_max = 100
        elif month < quarter * 3:
            new_season = Season.WINTER
            self.constrains.plants_max = 50
        else:
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
        stats_text = f"DATE: {self.statistics.day}/{self.statistics.month}/{self.statistics.year} | Plants: {self.statistics.plants} | Sheep: {self.statistics.sheep} | Foxes: {self.statistics.foxes} | Season : {self.statistics.season.value}"
        text_surface = font.render(stats_text, True, FONT_COLOR)
        screen.blit(text_surface, (10, 10))