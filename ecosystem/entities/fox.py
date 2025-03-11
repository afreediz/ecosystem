# entities/carnivores.py - Carnivore classes

import random
import math
import pygame
from entities.base import Entity
from entities.sheep import Sheep
from config.settings import WIDTH, HEIGHT, IMAGE_PATHS
from config.parameters import FOX_PARAMS

class Fox(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, IMAGE_PATHS['fox'], size=FOX_PARAMS['size'])
        self.energy = FOX_PARAMS['initial_energy']
        self.speed = random.uniform(*FOX_PARAMS['speed_range'])
        self.reproduction_threshold = FOX_PARAMS['reproduction_threshold']
        self.target = None
    
    def update(self, ecosystem):
        super().update()
        self.energy -= FOX_PARAMS['energy_consumption_rate']
        
        # Find food if no target or target is dead
        if self.target is None or not self.target.alive:
            self.find_food(ecosystem)
        
        # Move towards food
        if self.target:
            self.move_towards(self.target)
            
            # Check if close enough to eat
            if self.distance_to(self.target) < self.size/2 + self.target.size/2:
                self.eat()
        else:
            # Random movement if no food found
            self.x += random.uniform(-self.speed, self.speed)
            self.y += random.uniform(-self.speed, self.speed)
        
        # Keep within bounds
        self.x = max(0, min(WIDTH, self.x))
        self.y = max(0, min(HEIGHT, self.y))
        
        # Reproduction
        if self.energy > self.reproduction_threshold and random.random() < FOX_PARAMS['reproduction_chance']:
            self.reproduce(ecosystem)
    
    def find_food(self, ecosystem):
        closest_distance = float('inf')
        for entity in ecosystem.entities:
            if isinstance(entity, Sheep) and entity.alive:
                distance = self.distance_to(entity)
                if distance < closest_distance:
                    closest_distance = distance
                    self.target = entity
    
    def move_towards(self, target):
        dx = target.x - self.x
        dy = target.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            dx = dx / distance * self.speed
            dy = dy / distance * self.speed
            self.x += dx
            self.y += dy
    
    def eat(self):
        if self.target and self.target.alive:
            energy_gain = min(50, self.target.energy)
            self.energy += energy_gain
            self.target.alive = False
            self.target = None
    
    def reproduce(self, ecosystem):
        # Create a new fox nearby
        offset_x = random.randint(-30, 30)
        offset_y = random.randint(-30, 30)
        new_x = max(0, min(WIDTH, self.x + offset_x))
        new_y = max(0, min(HEIGHT, self.y + offset_y))
        
        new_fox = Fox(new_x, new_y)
        ecosystem.entities.append(new_fox)
        self.energy -= FOX_PARAMS['reproduction_cost']