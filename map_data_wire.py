import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    import pygame.draw
import numpy as np
import pickle

from dodecahedron import *
from projection import *



# Constants
WIDTH = 1000
HEIGHT = 1000
FPS = 15
LED_SCALE = 1
LEDS_PER_EDGE = 17



# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Infinity Dodecahedron Playground')
clock = pygame.time.Clock()


# Main loop prep
done = False
angle = 0
d = Dodecahedron(LEDS_PER_EDGE, mapped_edges=None)

data_wire = [d.edges[0]]
mapped_edges = [0]
green, red = d.edges[0].neighbors1

# Main loop
while not done:
    # Check events
    for e in pygame.event.get():
        if e.type == pygame.KEYDOWN:
            if e.key in [pygame.K_ESCAPE, pygame.K_q]:
                done = True
                break

            elif e.key == pygame.K_g and green is not None:
                # Add green edge
                if green.v0 not in [data_wire[-1].v0, data_wire[-1].v1]:
                    green.flip()
                data_wire.append(green)
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Stay at previous vertex
                    green = None
                else:
                    # Move forward along added edge and mark new directions
                    if green.neighbors1[1] not in data_wire:
                        red = green.neighbors1[1]
                    else:
                        red = None
                    if green.neighbors1[0] not in data_wire:
                        green = green.neighbors1[0]
                    else:
                        green = None
                mapped_edges.append(d.edges.index(data_wire[-1]))

            elif e.key == pygame.K_r and red is not None:
                # Add red edge
                if red.v0 not in [data_wire[-1].v0, data_wire[-1].v1]:
                    red.flip()
                data_wire.append(red)
                if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    # Stay at previous vertex
                    red = None
                else:
                    # Move forward along added edge and mark new directions
                    if red.neighbors1[0] not in data_wire:
                        green = red.neighbors1[0]
                    else:
                        green = None
                    if red.neighbors1[1] not in data_wire:
                        red = red.neighbors1[1]
                    else:
                        red = None
                mapped_edges.append(d.edges.index(data_wire[-1]))

    # Preparation
    clock.tick(FPS)
    screen.fill((0,0,0))
    d.turn_off()

    # Highlight mapped edges
    for e in data_wire:
        for led, shade in zip(e.leds, np.linspace(32,255,17)):
            led.color = np.array([int(shade)]*3)

    # Highlight next options
    if green is not None:
        for led, shade in zip(green.leds, np.linspace(128,255,17)):
            led.color = np.array([0, int(shade), 0])
    if red is not None:
        for led, shade in zip(red.leds, np.linspace(128,255,17)):
            led.color = np.array([int(shade), 0, 0])

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

    # Show new frame
    pygame.display.update()


print(mapped_edges)
pickle.dump(mapped_edges, open('mapped_edges.pkl', 'wb'))
