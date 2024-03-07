import numpy as np
# from pynput import keyboard
from time import time, sleep

import board
import neopixel

from dodecahedron import *
from projection import *



# Constants
FPS = 15
MS_PER_FRAME = 1000/FPS
BRIGHTNESS = 0.01
MAX_AMPERE = 5
LEDS_PER_EDGE = 17

# Set up data structure
print('Setting up data structure...')
d = Dodecahedron(LEDS_PER_EDGE, mapped_edges='mapped_edges.pkl')

# Set up LED strip
print('Setting up NeoPixel strip...')
pixels = neopixel.NeoPixel(board.D18, len(d.leds), brightness=BRIGHTNESS)

# # Set up keyboard listener
# print('Setting up keyboard listener...')
# key_queue = []
# def on_release(key):
#     global key_queue
#     key_queue.append(key)
# listener = keyboard.Listener(on_release=on_release, suppress=True)
# listener.start()

# Main loop prep
done = False
t_delta_ms = 0
times = []



# Main loop
print('Entering main loop...')
while not done:
    # Measure time
    t_prev = time()

    # # Handle keyboard events
    # if len(key_queue) > 0:
    #     key = key_queue.pop(0)
    #     if key == keyboard.Key.esc:
    #         done = True
    #         break
    #     elif key == keyboard.Key.left:
    #         d.next_animation(prev=True)
    #         times.clear()
    #         print(f'prev animation: {d.animation.__class__.__name__}', flush=True)
    #     elif key == keyboard.Key.right:
    #         d.next_animation(prev=False)
    #         times.clear()
    #         print(f'next animation: {d.animation.__class__.__name__}', flush=True)

    # Run animation
    try:
        d.animation.step(t_delta_ms)
    except:
        done = True
        print('Error :(')
        break

    # Lower brightness iff it exceeds set ampere threshold
    ampere = (np.sum(d.colors)/255) * 0.060 * BRIGHTNESS
    if ampere > MAX_AMPERE:
        pixels.brightness = BRIGHTNESS * MAX_AMPERE/ampere
    else:
        pixels.brightness = BRIGHTNESS

    # Send data to LED strip
    for i in range(len(d.leds)):
        pixels[i] = int(d.colors[i, 0]), int(d.colors[i, 1]), int(d.colors[i, 2])

    # Measure time
    t_delta_ms = (time() - t_prev) * 1000

    # Don't go faster than set FPS
    if t_delta_ms < MS_PER_FRAME:
        sleep((MS_PER_FRAME - t_delta_ms)/1000)

    # Console output
    times.append(t_delta_ms)
    if len(times) > 1:
        print(f'{1000/np.mean(times[-100:][1:]):.0f} FPS  |  {ampere:.1f} A', flush=True)


# Turn off LED strip
pixels.fill((0,0,0))
