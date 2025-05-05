# entities/base.py - Base Entity class
import random
import pygame
import math
from ecosystem.config.settings import FPS
class Entity:
    def __init__(self, x, y, image_path, size=20, gender=None, max_age=100):
        self.x = x
        self.y = y
        self.size = size
        self.energy = 100
        self.age = 0
        self.max_age = 100
        self.alive = True
        self.images_relative_path = 'assets/images/'
        self.gender = gender if gender is not None else random.randint(0, 1)  # 0 for female

        
        
        # Load and scale image
        self.original_image = pygame.image.load(self.images_relative_path + image_path)
        self.image = pygame.transform.scale(self.original_image, (size, size))
        self.rect = self.image.get_rect(center=(int(x), int(y)))

    def increment_age(self):
        """Increment the age of the entity"""
        self.age += 1
        if self.age > self.max_age:
            self.alive = False

    def update(self, stats):
        self.age += 1
        self.energy -= 0.1  # Basic energy consumption
        if stats.num_days % (3 * FPS) == 0:
            self.increment_age()
            
        if self.energy <= 0:
            self.alive = False

        if self.age > self.max_age:
            self.alive = False
        
        # Update rectangle position
        self.rect.center = (int(self.x), int(self.y))
    
    def draw(self, screen):
        # Draw entity at current position
        screen.blit(self.image, self.rect)
    
    def distance_to(self, other_entity):
        """Calculate distance to another entity"""
        return math.sqrt((self.x - other_entity.x)**2 + (self.y - other_entity.y)**2)