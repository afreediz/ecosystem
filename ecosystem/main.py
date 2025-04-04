# main.py - Entry point for the application

import pygame
import sys
from config.settings import WIDTH, HEIGHT, FPS, BACKGROUND_COLOR
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
    ecosystem.populate(num_plants=30, num_herbivores=15, num_carnivores=5)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                x, y = pygame.mouse.get_pos()
                
                if event.key == pygame.K_s:
                    ecosystem.entities.append(Sheep(x, y))
                if event.key == pygame.K_f:
                    ecosystem.entities.append(Fox(x, y))
                # Add a Sheep at mouse position when clicked
        
        ecosystem.update()
        ecosystem.draw(screen)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()