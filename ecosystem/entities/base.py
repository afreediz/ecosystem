# entities/base.py - Base Entity class

import pygame
import math
import random
import numpy as np
from config.settings import smallest_entity_size
from typing import TYPE_CHECKING
from collections import deque
from uuid import uuid4

if TYPE_CHECKING:
    from environment.ecosystem import Ecosystem

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
        self.id = uuid4().hex

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
    
# make brain, into body
class Brain:
    def __init__(self, id, body:'Animal') -> None:
        self.id = id
        self.body = body
        self.vision_range = body.vision_range
        
        # matrix width
        num_of_blocks = int((2 * self.vision_range) / smallest_entity_size)
        self.perception = np.zeros((num_of_blocks, num_of_blocks), dtype='object')
        self.memory = {}

        print(f"""BRAIN({id}) : 
            vision_range     : {body.vision_range}
            perception       : {self.perception.shape}""")

    def update_preception(self, x, y, ecosystem: 'Ecosystem') -> None:
        self.perception[:] = 0  # Optional: reset
        matrix = self.perception

        num_of_blocks = self.perception.shape[0]
        lower_x, lower_y = self.body.x - num_of_blocks*smallest_entity_size, self.body.y - num_of_blocks*smallest_entity_size
        
        for x in range(num_of_blocks):
            for y in range(num_of_blocks):
                current_x = lower_x + num_of_blocks*x
                current_y = lower_y + num_of_blocks*y

                matrix[x][y] = ecosystem.check_entity_presense(
                    bouding_box=(current_x, current_y, num_of_blocks, num_of_blocks), exclude_id=self.body.id
                )

        self.perception = matrix

    def show_perception(self) -> None:
        print(self.perception)

    def get_nearest_entity(self, entity_pereception_num:int):
        matrix = self.perception
        n = matrix.shape[0]
        
        # Find center position
        center = n // 2
        
        # Get all positions where value is 1
        pos = np.where(matrix == entity_pereception_num)        
        if len(pos[0]) == 0:
            return None  # Not found
        # self.show_perception()

        # Calculate distances from center to all 1s
        distances = []
        positions = []
        
        for i in range(len(pos[0])):
            row, col = pos[0][i], pos[1][i]
            # Calculate Euclidean distance from center
            distance = np.sqrt((row - center)**2 + (col - center)**2)
            distances.append(distance)
            positions.append((row, col))
        
        # Find position with minimum distance
        min_idx = np.argmin(distances)
        min_dist_pos = positions[min_idx]

        if not min_dist_pos:
            return None
        
        # signed entity position from matrice
        min_row, min_col = min_dist_pos
        if min_row < center:
            entity_x = -min_row*smallest_entity_size
        else:
            entity_x = min_row*smallest_entity_size

        if min_col < center:
            entity_y = -min_col*smallest_entity_size
        else:
            entity_y = min_col*smallest_entity_size

        # print(F"RETURNING {entity_x + (smallest_entity_size/2), entity_y + (smallest_entity_size/2)}")
        # return unit block center instead of top-left
        return entity_x + (smallest_entity_size/2), entity_y + (smallest_entity_size/2)

class Animal(Entity):
    def __init__(self, x, y, image_path, size=20):
        super().__init__(x, y, image_path, size)
        self.reproduction_threshold = 20
        self.vision_range = 100
        self.speed = 10
        self.target = None
        self.brain:Brain|None = None
        self.monitor_chance = 5

    def update(self, ecosystem:'Ecosystem'):
        super().update()

        if self.brain is not None:
            self.brain.update_preception(x=self.x, y=self.y, ecosystem=ecosystem)

            # if random.randint(1, 20) < 2:
            #     self.brain.show_perception()

    def _init(self):
        self._create_brain()

    def _create_brain(self):
        self.brain = Brain(body=self, id=self.id)
    
    def think_and_act(self):
        raise ValueError("Not implemented")

    def _train(self):
        raise ValueError("Not implemented")

    def _forward(self):
        raise ValueError("Not implemented")