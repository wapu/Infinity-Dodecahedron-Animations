import numpy as np
from time import time, sleep

# import board
# import neopixel

from dodecahedron import *
from projection import *



# Constants
FPS = 15
MS_PER_FRAME = 1000/FPS
LED_SCALE = 0.1
MAX_AMPERE = 10
LEDS_PER_EDGE = 17


d = Dodecahedron(LEDS_PER_EDGE, mapped_edges='mapped_edges.pkl')


# Set up LED strip
# pixels = neopixel.NeoPixel(board.D18, len(d.leds), brightness=LED_SCALE)


# Main loop prep
done = False
t_delta_ms = 0
times = []
# peak_power = 0



# Main loop
while not done:
    # Measure time
    t_prev = time()

    # Check events - TODO find out how to steer
        # done = True
        # d.next_animation(prev=False)
        # times.clear()
        # peak_power = 0

    # Animate LEDs
    d.animation.step(t_delta_ms)

    # Clamp power usage and send data out
    ampere = (np.sum(d.colors)/255) * 0.060 * LED_SCALE
    if ampere > MAX_AMPERE:
        # pixels.brightness = LED_SCALE * ampere/MAX_AMPERE
        pass
    else:
        # pixels.brightness = LED_SCALE
        pass
    # for i in range(len(d.leds)):
    #     pixels[i] = int(d.colors[i, 0]), int(d.colors[i, 1]), int(d.colors[i, 2])

    # Measure time
    t_delta_ms = (time() - t_prev) * 1000
    times.append(t_delta_ms)

    # Maintain max FPS
    if t_delta_ms < MS_PER_FRAME:
        sleep((MS_PER_FRAME - t_delta_ms)/1000)

    # Console output
    # power = np.mean(d.colors / 255)
    # peak_power = max(power, peak_power)
    # f'{power*100:02.0f}% of max RGB usage (peak {peak_power*100:.0f}%)'
    # f'{power*60*30*LEDS_PER_EDGE*LED_SCALE/1000:.1f} Ampere at {LED_SCALE*100:.0f}% LED power (peak {peak_power*60*30*LEDS_PER_EDGE*LED_SCALE/1000:.1f}A)'
    print(f'{1000/np.mean(times[-100:]):.0f} FPS', flush=True)
    # f'Animation {d.animation_index+1}/{len(ANIMATION_CYCLE)}: "{d.animation.__class__.__name__}"'
