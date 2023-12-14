import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    import pygame.draw
import numpy as np
from time import time

from dodecahedron import *
from projection import *



# Constants
WIDTH = 1000
HEIGHT = 1000
FPS = 15
LED_SCALE = .4
LEDS_PER_EDGE = 17


# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Infinity Dodecahedron Playground')
clock = pygame.time.Clock()
pygame.font.init()
font = pygame.font.SysFont('CMU Sans Serif', 30)


# Main loop prep
done = False
pause = False
angle = 0
d = Dodecahedron(LEDS_PER_EDGE)


# Main loop
while not done:
    # Check events
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            if e.key in [pygame.K_ESCAPE, pygame.K_q]:
                done = True
                break
            elif e.key == pygame.K_SPACE:
                pause = not pause
            else:
                # Cycle through animations
                d.next_animation()

    # Preparation
    t_delta_ms = clock.tick(FPS)
    if pause:
        continue
    screen.fill((0,0,0))

    # Animate LEDs
    d.animation.step(t_delta_ms)

    # Rotate camera matrix
    angle += 0.001 * np.pi
    camera_matrix = get_camera_matrix(theta_z=0.5*angle, theta_y=angle, z=3)

    # Project and draw dodecahedron
    # Edges
    for e in d.edges:
        (v0, v1), (z0, z1) = project(camera_matrix, [e.v0.pos, e.v1.pos], (WIDTH, HEIGHT))
        pygame.draw.line(screen, (5,5,5), v0, v1, width=int(15/(z0+z1)))
    # LEDs sorted and scaled by distance
    leds, dists = project(camera_matrix, d.get_leds(), (WIDTH, HEIGHT))
    leds = sorted(zip(leds, dists, d.get_colors()), key=lambda pzc: pzc[1])
    for led, dist, color in leds:
        pygame.draw.circle(screen, color, led, radius=6 * 1/dist)

    # Display power consumption
    power = np.mean(d.get_colors()/255)
    screen.blit(font.render(f'{power*100:.0f}% of max RGB usage', False, (255,255,255)), (10, 10))
    screen.blit(font.render(f'{power*60*30*LEDS_PER_EDGE*LED_SCALE/1000:.1f} Ampere at {LED_SCALE*100:.0f}% LED power', False, (255,255,255)), (10, 45))
    screen.blit(font.render(f'Animation: "{d.animation.__class__.__name__}"', False, (255,255,255)), (10, HEIGHT - 40))

    # Show new frame
    pygame.display.update()
