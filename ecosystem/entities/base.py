# entities/base.py - Base Entity class

import pygame
import math
import random
import numpy as np
from config.settings import largest_entity_size
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
        self.vision_block_size = int(4*body.vision_range**2 / largest_entity_size**2)
        self.block_length = int(math.sqrt(self.vision_block_size))
        self.preception = np.zeros(self.vision_block_size)
        self.memory = {}

        print(f"""BRAIN({id}) : 
            vision_range     : {body.vision_range}
            vision_block_size: {self.vision_block_size}
            block_length     : {self.block_length}
            perception       : {len(self.preception)}""")

    def update_preception(self, x, y, ecosystem: 'Ecosystem') -> None:
        half_vision = self.vision_range
        grid_size = self.block_length  # number of blocks per row/col (square grid)
        step = (2 * self.vision_range) / grid_size  # size of one block in pixels

        self.perception = np.zeros(self.vision_block_size)  # Optional: reset

        preception_index = 0
        for row in range(grid_size):
            for col in range(grid_size):
                # Compute the top-left corner of the current block
                box_x = int(x - half_vision + col * step)
                box_y = int(y - half_vision + row * step)
                bounding_box = (box_x, box_y, step, step)

                if preception_index < len(self.perception):
                    self.perception[preception_index] = ecosystem.check_entity_presense(
                        bouding_box=bounding_box, exclude_id=self.id
                    )
                    preception_index += 1

    def show_perception(self, external_preception=None) -> None:
        perception = external_preception if external_preception else self.perception
        size = len(perception)
        width = int(math.sqrt(size))

        if width * width != size:
            width += 1  # Try to accommodate overflow visually

        if external_preception is not None:
            print("DUMMY")
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

    # def get_nearest_entity(self, entity_type:Entity):
    #     pos_x, pos_y = self.body.x, self.body.y

    #     # radial outward check for given entity
    #     size = len(self.preception)
    #     width = int(math.sqrt(size))
    #     radial_width = int(width / 2)
    #     block_length = self.block_length

    #     for i in range(radial_width):
    #         lower_x, lower_y = pos_x - self.block_length * i, pos_y - self.block_length * i
    #         upper_x, upper_y = pos_x + self.block_length * i, pos_y + self.block_length * i

    #         for j in range(width):
    #             idx = j + i * width

    def is_entity_near(self, entity_preception_number: int) -> int|None:
        """
        Find the nearest entity from the center of the flattened matrix using BFS.
        
        Args:
            entity_preception_number: The value to search for in the matrix
            
        Returns:
            int: 1D index of nearest entity, or None if not found
        """
        matrix = self.preception
        size = len(matrix)
        width = int(math.sqrt(size))
        rows = cols = width
        
        # Helper functions to convert between 1D and 2D indices
        def to_1d_index(row, col):
            return row * width + col
        
        # Find center position
        center_row = rows // 2
        center_col = cols // 2
        
        # BFS setup
        queue = deque([(center_row, center_col, 0)])  # (row, col, distance)
        visited = set()
        visited.add((center_row, center_col))

        dummy_visited = [0 for _ in range(len(matrix))]
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1), (1, -1), (-1,1), (-1, -1)]
        
        while queue:
            row, col, dist = queue.popleft()
            
            # Check all 4 directions
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                
                # Check bounds
                if 0 <= new_row < rows and 0 <= new_col < cols:
                    if (new_row, new_col) not in visited:
                        visited.add((new_row, new_col))
                        
                        # Convert to 1D index to access flattened matrix
                        flat_index = to_1d_index(new_row, new_col)
                        
                        # If we found the entity, return its 1D index
                        dummy_visited[flat_index] = 1
                        if matrix[flat_index] == entity_preception_number:
                            return flat_index
                        
                        # Add to queue for further exploration
                        queue.append((new_row, new_col, dist + 1))
        
        # No entity found in the matrix
        self.show_perception(external_preception=dummy_visited)
        return None


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