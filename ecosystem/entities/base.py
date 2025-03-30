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
        
        # Load and scale image
        self.original_image = pygame.image.load(self.images_relative_path + image_path)
        self.image = pygame.transform.scale(self.original_image, (size, size))
        self.rect = self.image.get_rect(center=(int(x), int(y)))
    
    def update(self):
        self.age += 1
        self.energy -= 0.1  # Basic energy consumption
        if self.energy <= 0:
            self.alive = False
        
        # Update rectangle position
        self.rect.center = (int(self.x), int(self.y))
    
    def draw(self, screen):
        # Draw entity at current position
        screen.blit(self.image, self.rect)
    
    def distance_to(self, other_entity):
        """Calculate distance to another entity"""
        return math.sqrt((self.x - other_entity.x)**2 + (self.y - other_entity.y)**2)