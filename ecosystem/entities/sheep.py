# entities/herbivores.py - Herbivore classes

import random
import math
import pygame
from entities.base import Entity
from entities.plant import Plant
from config.settings import WIDTH, HEIGHT, IMAGE_PATHS
from config.parameters import SHEEP_PARAMS
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.ecosystem import Ecosystem

class Sheep(Entity):
    def __init__(self, x, y):
        super().__init__(x, y, IMAGE_PATHS.sheep, size=SHEEP_PARAMS.size)
        self.energy = SHEEP_PARAMS.initial_energy
        self.speed = random.uniform(*SHEEP_PARAMS.speed_range)
        self.reproduction_threshold = SHEEP_PARAMS.reproduction_threshold
        self.target = None
        self.vision_range = 70
        self.name = 'sheep'
    
    def update(self, ecosystem:'Ecosystem'):
        super().update()
        self.energy -= SHEEP_PARAMS.energy_consumption_rate
        
        predator = self.find_nearest_predator(ecosystem=ecosystem)
        
        if predator:
            self.run_from_predators(target=predator)
        else:
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
        if self.energy > self.reproduction_threshold and random.random() < SHEEP_PARAMS.reproduction_chance:
            if ecosystem.statistics.sheep < ecosystem.constrains.sheeps_max:
                self.reproduce(ecosystem)
    
    def find_food(self, ecosystem):
        closest_distance = float('inf')
        for entity in ecosystem.entities:
            if isinstance(entity, Plant) and entity.alive:
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
            energy_gain = min(25, self.target.energy)
            self.energy += energy_gain
            self.target.energy -= energy_gain
            
            if self.target.energy <= 0:
                self.target.alive = False
            
            self.target = None
    
    def reproduce(self, ecosystem):
        # Create a new sheep nearby
        offset_x = random.randint(-20, 20)
        offset_y = random.randint(-20, 20)
        new_x = max(0, min(WIDTH, self.x + offset_x))
        new_y = max(0, min(HEIGHT, self.y + offset_y))
        
        new_sheep = Sheep(new_x, new_y)
        ecosystem.entities.append(new_sheep)
        self.energy -= SHEEP_PARAMS.reproduction_cost

    def run_from_predators(self, target):
        dx = target.x - self.x
        dy = target.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance > 0:
            dx = dx / distance * self.speed
            dy = dy / distance * self.speed
            self.x -= dx
            self.y -= dy  

    def find_nearest_predator(self, ecosystem:'Ecosystem'):
        closest_distance = float('inf')
        predator = None
        for entity in ecosystem.entities:
            if entity.name == 'fox' and entity.alive:
                distance = self.distance_to(entity)
                if distance < self.vision_range:
                    if distance < closest_distance:
                        closest_distance = distance
                        predator = entity

        return predator