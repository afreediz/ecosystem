# entities/base.py - Base Entity class

import pygame
import math

class Entity:
    def __init__(self, x, y, image_path, size=20):
        self.x = x
        self.y = y
        self.size = size
        self.energy = 100
        self.age = 0
        self.alive = True
        self.images_relative_path = 'assets/images/'
        self.energy_consumption_rate = 0.1

        # Load and scale image
        self.original_image = pygame.image.load(self.images_relative_path + image_path)
        self.image = pygame.transform.scale(self.original_image, (size, size))
        self.rect = self.image.get_rect(center=(int(x), int(y)))
    
    def update(self):
        self.age += 1
        self.energy -= self.energy_consumption_rate
        if self.energy <= 0:
            self.alive = False
        
        # Update rectangle position
        self.rect.center = (int(self.x), int(self.y))
    
    def draw(self, screen:pygame.Surface):
        # Draw entity at current position
        screen.blit(self.image, self.rect)
    
    def distance_to(self, other_entity):
        """Calculate distance to another entity"""
        return math.sqrt((self.x - other_entity.x)**2 + (self.y - other_entity.y)**2)
    
class Animal(Entity):
    def __init__(self, x, y, image_path, size=20):
        super().__init__(x, y, image_path, size)
        self.reproduction_threshold = 20
        self.vision_range = 30
        self.speed = 10
        self.target = None
        self.brain = self._get_brain()

    def _get_brain(self):
        pass
    
    def think_and_act(self):
        pass