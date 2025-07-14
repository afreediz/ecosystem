# entities/base.py - Base Entity class

import pygame
import math
import random
import numpy as np
from config.settings import largest_entity_size
from typing import TYPE_CHECKING

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
        self.id = random.randint(0, 1000)

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
    
class Brain:
    def __init__(self, id, vision_range) -> None:
        self.id = id
        self.vision_range = vision_range
        self.vision_block_size = int(4*vision_range**2 / largest_entity_size**2)
        self.block_length = int(math.sqrt(self.vision_block_size))
        self.preception = np.zeros(self.vision_block_size)
        self.memory = {}

        print(f"""BRAIN({id}) : 
            vision_range     : {vision_range}
            vision_block_size: {self.vision_block_size}
            block_length     : {self.block_length}
            perception       : {len(self.preception)}""")

    def update_preception(self, x, y, ecosystem: 'Ecosystem') -> None:
        half_vision = self.vision_range
        grid_size = self.block_length  # number of blocks per row/col (square grid)
        step = (2 * self.vision_range) / grid_size  # size of one block in pixels

        self.perception = np.zeros(self.vision_block_size)  # Optional: reset

        preception_index = 0
        detected = False
        for row in range(grid_size):
            for col in range(grid_size):
                # Compute the top-left corner of the current block
                box_x = int(x - half_vision + col * step)
                box_y = int(y - half_vision + row * step)
                bounding_box = (box_x, box_y, step, step)

                if preception_index < len(self.perception):
                    res = ecosystem.check_entity_presense(
                        bouding_box=bounding_box, exclude_id=self.id
                    )
                    self.perception[preception_index] = res
                    if res > 0 :
                        detected = True
                    preception_index += 1

        if detected:
            self.show_perception()

    def show_perception(self) -> None:
        perception = self.perception
        size = len(perception)
        width = int(math.sqrt(size))

        if width * width != size:
            print(f"[Warning] Perception size ({size}) is not a perfect square.")
            width += 1  # Try to accommodate overflow visually

        print(f"Perception Grid ({width} x {width}):")
        for i in range(width):
            for j in range(width):
                idx = j + i * width
                if idx < size:
                    value = perception[idx]
                    # Optional: format value as int if it's categorical (e.g., 0=empty, 1=plant, 2=fox)
                    print(f"{int(value):2}", end=" ")
                else:
                    print(" .", end=" ")  # Empty cell for overflow
            print()  # Newline after each row

class Animal(Entity):
    def __init__(self, x, y, image_path, size=20):
        super().__init__(x, y, image_path, size)
        self.reproduction_threshold = 20
        self.vision_range = 100
        self.speed = 10
        self.target = None
        self.brain:Brain|None = None

    def update(self, ecosystem:'Ecosystem'):
        super().update()

        if self.brain is not None:
            self.brain.update_preception(x=self.x, y=self.y, ecosystem=ecosystem)
            
            # if max(self.brain.preception) > 0:
            #     self.brain.show_perception()

    def _init(self):
        self._create_brain()

    def _create_brain(self):
        self.brain = Brain(vision_range=self.vision_range, id=self.id)
    
    def think_and_act(self):
        raise ValueError("Not implemented")

    def _train(self):
        raise ValueError("Not implemented")

    def _forward(self):
        raise ValueError("Not implemented")