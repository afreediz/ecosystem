# entities/carnivores.py - Carnivore classes

import random
import math
from entities.base import Entity, Animal
from config.settings import WIDTH, HEIGHT, IMAGE_PATHS
from config.parameters import FOX_PARAMS
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment.ecosystem import Ecosystem
    from entities.sheep import Sheep

class Fox(Animal):
    def __init__(self, x, y):
        super().__init__(x, y, IMAGE_PATHS.fox, size=FOX_PARAMS.size)
        self.energy = FOX_PARAMS.initial_energy
        self.speed = FOX_PARAMS.speed
        self.reproduction_threshold = FOX_PARAMS.reproduction_threshold
        self.energy_consumption_rate = FOX_PARAMS.energy_consumption_rate
        self.vision_range = FOX_PARAMS.vision_range
        self.name = 'fox'

        self._init()
    
    def update(self, ecosystem:'Ecosystem'):
        super().update(ecosystem=ecosystem)

        # make previous target None if its dead
        if self.target is not None and self.target.alive is False:
            self.target = None

        target = self.find_food(ecosystem)

        if self.target is not None and target is not None and self.target != target:
            # if already target exist, but new target is much closer by priority
            current_priority = 3
            if self.distance_to(self.target) > self.distance_to(target)* (1/current_priority):
                self.target = target
        else:
            self.target = target

        # Move towards food
        if self.target:
            self.move_towards(self.target) #type:ignore
            
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
        if self.energy > self.reproduction_threshold and random.random() < FOX_PARAMS.reproduction_chance:
            if ecosystem.statistics.foxes < ecosystem.constrains.fox_max:
                self.reproduce(ecosystem)
    
    def find_food(self, ecosystem:'Ecosystem'):
        closest_distance = float('inf')

        target = None
        for entity in ecosystem.entities:
            if entity.name == 'sheep' and entity.alive:
                distance = self.distance_to(entity)
                if distance < self.vision_range:
                    if distance < closest_distance:
                        closest_distance = distance
                        target = entity
        
        return target

    def move_towards(self, target:'Sheep'):
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
            self.speed += 0.1
            self.vision_range += 3
    
    def reproduce(self, ecosystem:'Ecosystem'):
        # Create a new fox nearby
        offset_x = random.randint(-30, 30)
        offset_y = random.randint(-30, 30)
        new_x = max(0, min(WIDTH, self.x + offset_x))
        new_y = max(0, min(HEIGHT, self.y + offset_y))
        
        new_fox = Fox(new_x, new_y)
        ecosystem.entities.append(new_fox)
        self.energy -= FOX_PARAMS.reproduction_cost