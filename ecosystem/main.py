# main.py - Entry point for the application

import pygame
import sys
import random
from config.settings import WIDTH, HEIGHT, FPS, BACKGROUND_COLOR
from entities.plant import Plant
from entities.sheep import Sheep
from entities.fox import Fox
from environment.ecosystem import Ecosystem

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ecosystem Simulation")
clock = pygame.time.Clock()

def main():
    # Create and populate ecosystem
    ecosystem = Ecosystem()
    ecosystem.populate(num_plants=30, num_herbivores=10, num_carnivores=5)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Add a plant at mouse position when clicked
                x, y = pygame.mouse.get_pos()
                ecosystem.entities.append(Plant(x, y))
        
        ecosystem.update()
        ecosystem.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()