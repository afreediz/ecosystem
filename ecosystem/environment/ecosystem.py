# environment/ecosystem.py - Ecosystem class

import random
import pygame
from config.settings import WIDTH, HEIGHT, FONT_NAME, FONT_SIZE, FONT_COLOR, BACKGROUND_COLOR
from entities.plant import Plant
from entities.sheep import Sheep
from entities.fox import Fox

class Ecosystem:
    def __init__(self):
        self.entities = []
        self.statistics = {"day": 0, "plants": 0, "sheep": 0, "foxes": 0}
    
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
        self.statistics["day"] += 1
        
        # Update all entities
        for entity in self.entities:
            if entity.alive:
                entity.update(self)
        
        # Remove dead entities
        self.entities = [entity for entity in self.entities if entity.alive]
        
        # Update statistics
        self.statistics["plants"] = sum(1 for entity in self.entities if isinstance(entity, Plant))
        self.statistics["sheep"] = sum(1 for entity in self.entities if isinstance(entity, Sheep))
        self.statistics["foxes"] = sum(1 for entity in self.entities if isinstance(entity, Fox))
        
        # Add some random plants occasionally
        if random.random() < 0.1:
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.entities.append(Plant(x, y))
    
    def draw(self, screen):
        # Draw background
        screen.fill(BACKGROUND_COLOR)
        
        # Draw all entities
        for entity in self.entities:
            entity.draw(screen)
        
        # Draw statistics
        font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        stats_text = f"Day: {self.statistics['day']} | Plants: {self.statistics['plants']} | Sheep: {self.statistics['sheep']} | Foxes: {self.statistics['foxes']}"
        text_surface = font.render(stats_text, True, FONT_COLOR)
        screen.blit(text_surface, (10, 10))