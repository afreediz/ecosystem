# entities/herbivores.py - Herbivore classes

import random
import math
import pygame
from entities.base import Animal
from entities.plant import Plant
from config.settings import WIDTH, HEIGHT, IMAGE_PATHS
from config.parameters import SHEEP_PARAMS, ENTITY_IN_PRECEPTION
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.ecosystem import Ecosystem
    from entities.plant import Plant
    from entities.fox import Fox

class Sheep(Animal):
    def __init__(self, x, y, debug=False):
        super().__init__(x, y, IMAGE_PATHS.sheep, size=SHEEP_PARAMS.size)
        self.energy = SHEEP_PARAMS.initial_energy
        self.speed = SHEEP_PARAMS.speed
        self.reproduction_threshold = SHEEP_PARAMS.reproduction_threshold
        self.vision_range = SHEEP_PARAMS.vision_range
        self.energy_consumption_rate = SHEEP_PARAMS.energy_consumption_rate
        self.debug = debug
        self.name = 'sheep'

        self._init()
    
    def update(self, ecosystem:'Ecosystem'):
        super().update(ecosystem=ecosystem)

        predator = self.find_nearest_predator(ecosystem=ecosystem)

        if predator:
            self.run_from_predators(target=predator) #type:ignore
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

        if random.randint(1, 100) < self.monitor_chance:
            if self.brain is not None:
                self.brain.show_perception()
                res = self.brain.is_entity_near(entity_preception_number=ENTITY_IN_PRECEPTION.plant)
                print('PLANT FOUND AT : ', res)
    
    def find_food(self, ecosystem:'Ecosystem'):
        closest_distance = float('inf')
        for entity in ecosystem.entities:
            if isinstance(entity, Plant) and entity.alive:
                distance = self.distance_to(entity)
                if distance < closest_distance:
                    closest_distance = distance
                    self.target = entity
    
    def move_towards(self, target:'Plant'):
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
    
    def reproduce(self, ecosystem:'Ecosystem'):
        # Create a new sheep nearby
        offset_x = random.randint(-20, 20)
        offset_y = random.randint(-20, 20)
        new_x = max(0, min(WIDTH, self.x + offset_x))
        new_y = max(0, min(HEIGHT, self.y + offset_y))
        
        new_sheep = Sheep(new_x, new_y)
        ecosystem.entities.append(new_sheep)
        self.energy -= SHEEP_PARAMS.reproduction_cost

    def run_from_predators(self, target:'Fox'):
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