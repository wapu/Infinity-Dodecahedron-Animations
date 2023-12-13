import numpy as np
from colorsys import hls_to_rgb
from time import time
from collections import deque


# Base class, maybe unneccessary
class Animation():

    def __init__(self, dodecahedron, backlight=(0.,0.,0.)):
        self.d = dodecahedron

    def initialize(self):
        for led in self.d.leds:
            led.turn_off()

    def random_rgb(self):
        color = np.random.random_sample(3)**2
        color = 255 * color / np.max(color)
        return color


# General moving particle with fading trail
class Particle():

    # Life stages
    ALIVE = 0 # Moving at full power
    DYING = 1 # Petering out
    DEAD  = 2 # Only trail left
    GONE  = 3 # Nothing left, can delete

    def __init__(self, start, color, n_trailing, trail_style='linear', duration=None, end_style='linear'):
        # Starting position
        self.led = start
        # Neighboring LEDs the particle can't move to
        self.blocked = [np.random.choice(start.neighbors)]
        # Starting color
        self.color = np.array(color)

        # Trail stuff
        self.n_trailing = n_trailing
        self.trail_style = trail_style
        self.trail = []
        self.color_decrement = self.color / (self.n_trailing + 1)
        self.trail_factor = 0.01 ** (1/self.n_trailing + 1)
        self.erosion = 1 / (self.n_trailing + 1)

        # Duration stuff
        self.duration = duration
        self.state = self.ALIVE
        self.end_style = end_style

    def step(self, t_delta_ms=0):
        # Update trail
        if self.n_trailing > 0:
            if self.trail_style == 'erode':
                # Keep only the LEDs in trail that are still active
                self.trail = [self.led] + [led for led in self.trail if led.color.sum() > 0]
            else:
                # Truncate trail to fixed length
                self.trail = ([self.led] + self.trail)[:self.n_trailing]
        else:
            # No trail, turn off LED at current location
            self.led.turn_off()

        # Dim trail
        for led in self.trail:
            if self.trail_style == 'exponential':
                led.set_color(led.color * self.trail_factor)
            elif self.trail_style == 'linear':
                led.set_color(np.maximum(0, led.color - self.color_decrement))
            elif self.trail_style == 'erode':
                if np.random.random_sample() <= self.erosion:
                    led.turn_off()

        # Handle duration and end of life
        if self.duration is not None:
            if self.duration > 0:
                self.duration -= 1
            else:
                if self.state in [self.ALIVE, self.DYING]:
                    if self.end_style == 'sudden':
                        self.state = self.DEAD
                    elif self.end_style == 'linear':
                        self.state = self.DYING
                        self.color = np.maximum(0, self.color - self.color_decrement/2)
                        if self.color.sum() == 0:
                            self.state = self.DEAD
                if self.state == self.DEAD:
                    if np.sum([led.color for led in self.trail]) == 0:
                        self.state = self.GONE


        # Move particle further along LED graph
        if self.state in [self.ALIVE, self.DYING]:
            next_options = [n for n in self.led.neighbors if n not in self.blocked]
            next = np.random.choice(next_options)
            self.blocked = next_options + [self.led]
            self.led = next
            self.update_color()
            self.led.set_color(np.minimum(255, self.led.color + self.color))


    def update_color(self):
        pass


# Various animations
class TrailingSparks(Animation):

    def __init__(self, dodecahedron, n_sparks=30):
        super().__init__(dodecahedron)

        # Create sparks
        self.sparks = []
        for i in range(n_sparks):
            start = np.random.choice(self.d.edges).leds[self.d.leds_per_edge//2]
            self.sparks.append(
                    Particle(start, self.random_rgb(), 10, trail_style='linear')
                )

    def step(self, t_delta_ms=0):
        for spark in self.sparks:
            spark.step()


class RainbowWorms(Animation):
    pass


class LightUpFaces(Animation):

    def __init__(self, dodecahedron, speed=0.5, cooldown=9):
        super().__init__(dodecahedron)
        self.speed = speed
        self.cooldown = cooldown
        self.timer = 0.0

        self.active = []
        self.inactive = list(self.d.faces)

    def step(self, t_delta_ms=0):
        self.timer += 1
        if self.timer > 1/self.speed:
            self.timer -= 1/self.speed

            np.random.shuffle(self.inactive)
            activated = self.inactive[0]
            self.inactive = self.inactive[1:]

            for face in self.d.faces:
                face.color *= 0.5
            activated.color = self.random_rgb()
            self.active.append(activated)

            self.d.turn_off()
            for face in self.active:
                for led in face.leds:
                    led.set_color(np.minimum(255, led.color + face.color))

            if len(self.active) > self.cooldown:
                self.inactive.append(self.active[0])
                self.active = self.active[1:]


class ClosedLoop(Animation):
    pass


class PulsingVertices(Animation):
    pass


class Lightning(Animation):
    pass


class Glitter(Animation):

    def __init__(self, dodecahedron, n_per_step=15, afterglow=0.8, color=(255,255,255)):
        super().__init__(dodecahedron)
        self.n_per_step = n_per_step
        self.afterglow = afterglow
        self.color = None if color is None else np.array(color)

    def step(self, t_delta_ms=0):
        # Dim previous lights
        for led in self.d.leds:
            led.set_color(led.color * self.afterglow)
        # Spawn new glitter lights
        for i in range(self.n_per_step):
            if self.color is None:
                color = np.random.random_sample(3)**2
                color = 255 * color / np.max(color)
            else:
                color = self.color
            led = np.random.choice(self.d.leds)
            led.set_color(np.minimum(255, led.color + color))


class WanderingLanterns(Animation):
    pass


class RollingHue(Animation):

    def __init__(self, dodecahedron, speed=2.0):
        super().__init__(dodecahedron)
        self.speed = speed

    def step(self, t_delta_ms=0):
        for led in self.d.leds:
            # hue = np.cos(2*led.theta + led.phi) / 2 + 0.5
            hue = np.sin(led.theta + self.speed * time()) / 2 + 0.5
            # hue = np.cos(2*led.theta + 3*time()) / 2 + 0.5
            # hue = np.cos(2*led.theta + led.phi + 3*time()) / 2 + 0.5
            led.color = np.array(hls_to_rgb(hue, 0.5, 1.0)) * 255


ANIMATION_CYCLE = [
    LightUpFaces,
    RollingHue,
    TrailingSparks,
    Glitter,
]
