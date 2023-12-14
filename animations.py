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

    def random_hue(self):
        return self.hls_to_rgb(np.random.rand())

    def hls_to_rgb(self, hue, lightness=0.5, saturation=1.0):
        return np.array(hls_to_rgb(hue, lightness, saturation)) * 255

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
        self.color_decrement = self.color / (self.n_trailing)
        self.trail_factor = 0.01 ** (1/(self.n_trailing + 1))
        self.erosion = 1 / (self.n_trailing + 1)

        # Duration stuff
        self.duration = duration
        self.state = self.ALIVE
        self.end_style = end_style

        # For swarm behavior
        self.swarm = None

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
            if self.swarm is None or len(next_options) == 1:
                next = np.random.choice(next_options)
            else:
                d_swarm = [(np.linalg.norm(self.led.pos - p.led.pos) + 0.001*np.random.rand(), p) for p in self.swarm if p is not self]
                nearest_pos = sorted(d_swarm)[0][1].led.pos
                d_options = [(np.linalg.norm(led.pos - nearest_pos) + 0.001*np.random.rand(), led) for led in next_options]
                next = sorted(d_options)[-1][1]
            self.blocked = next_options + [self.led]
            self.led = next
            self.led.set_color(np.minimum(255, self.led.color + self.color))


# Various animations
class TrailingSparks(Animation):

    def __init__(self, dodecahedron, n_sparks=30, trail_style='linear'):
        super().__init__(dodecahedron)

        # Create sparks
        self.sparks = []
        for i in range(n_sparks):
            # start = np.random.choice(self.d.edges).leds[self.d.leds_per_edge//2]
            start = np.random.choice(self.d.leds)
            spark = Particle(start, self.random_rgb(), 10, trail_style=trail_style)
            self.sparks.append(spark)

    def step(self, t_delta_ms=0):
        for spark in self.sparks:
            spark.step()


class RainbowWorms(Animation):

    def __init__(self, dodecahedron, n_worms=8):
        super().__init__(dodecahedron)
        self.worm_length = 5 * self.d.leds_per_edge
        self.rainbow = [self.hls_to_rgb(h) for h in np.linspace(0, 1, self.worm_length)]

        # Create worms
        self.worms = []
        for i in range(n_worms):
            # start = np.random.choice(self.d.edges).leds[self.d.leds_per_edge//2]
            start = np.random.choice(self.d.leds)
            worm = Particle(start, (0,0,0), self.worm_length, trail_style='exponential')
            worm.rainbow_index = np.random.randint(len(self.rainbow))
            self.worms.append(worm)
        for worm in self.worms:
            worm.swarm = self.worms

    def step(self, t_delta_ms=0):
        for worm in self.worms:
            worm.rainbow_index = (worm.rainbow_index + 1) % len(self.rainbow)
            worm.color = self.rainbow[worm.rainbow_index]
            worm.step()


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


class PulsingCorners(Animation):

    class Corner():

        def __init__(self, vertex, speed=5, phases=True):
            self.leds = [e.leds if (e.v0 is vertex) else e.leds[::-1] for e in vertex.edges]
            self.speed = speed + 0.01*speed*np.random.randn()
            if phases:
                self.phase = np.random.rand() * 2 * np.pi
            else:
                self.phase = 0
            self.reroll()

        def reroll(self):
            self.hue = np.random.rand()
            self.extents = [np.random.randint(2, len(self.leds[0])//2 + 3) for l in self.leds]

    def __init__(self, dodecahedron, speed=5):
        super().__init__(dodecahedron)
        self.corners = [self.Corner(v, speed) for v in self.d.vertices]

    def step(self, t_delta_ms=0):
        self.d.turn_off()
        for c in self.corners:
            factor = (1 + np.sin(c.speed * time() + c.phase))/2
            if factor < 2/self.d.leds_per_edge:
                c.reroll()
            for leds, extent in zip(c.leds, c.extents):
                current_extent = int(extent * factor)
                for i in range(current_extent):
                    l = 0.25 + (i/current_extent)/2
                    s = 0.25 + 0.75 * (i/current_extent)
                    leds[i].set_color(self.hls_to_rgb(c.hue, l, s))


class Lightning(Animation):

    class Strike():
        def __init__(self, leds, sequence):
            self.state = 0
            self.leds = leds
            self.sequence = sequence

    def __init__(self, dodecahedron, likelihood=1, afterglow=0.9):
        super().__init__(dodecahedron)
        self.likelihood = likelihood
        self.afterglow = afterglow

        self.strikes = []
        self.timer = 5

    def step(self, t_delta_ms=0):
        # Create new lightning strikes at stochastic intervals
        self.timer += t_delta_ms/1000
        if self.timer > 1 + 10*np.random.rand():
            self.timer = 0

            # Grow branches from random point of impact
            origin = np.random.choice(self.d.leds)
            leds = []
            frontier = deque([(origin, 0),])
            while len(frontier) > 0:
                led, dist = frontier.popleft()
                if (led not in leds) and (dist < 5*self.d.leds_per_edge) and (np.random.rand() > 0.002*dist):
                    leds.append(led)
                    for n in led.neighbors:
                        frontier.append((n, dist+1))

            # Create random flashing sequence
            duration = np.random.randint(3, 6 + int(np.sqrt(len(leds))))
            sequence = np.maximum(0, (np.random.rand(duration) > .6) - np.random.rand(duration)**2)
            if sequence.sum() == 0:
                sequence[0] = 1
            sequence /= np.max(sequence)
            sequence[-1] = 0.3

            self.strikes.append(self.Strike(leds, sequence))

        # Dim previous lights
        for led in self.d.leds:
            led.set_color(led.color * self.afterglow)

        # Display lightning
        for strike in self.strikes:
            for led in strike.leds:
                led.set_color([255 * strike.sequence[strike.state]]*3)
            strike.state += 1
        self.strikes = [s for s in self.strikes if s.state < len(s.sequence)]


class CornerFireworks(Animation):

    def __init__(self, dodecahedron, interval=.2, duration_between=(7,13), trail_style='linear'):
        super().__init__(dodecahedron)
        self.interval = interval
        self.timer = 0
        self.duration_between = duration_between
        self.trail_style = trail_style
        self.sparks = []

    def step(self, t_delta_ms=0):
        # Update existing sparks
        self.sparks = [s for s in self.sparks if s.state != Particle.GONE]
        for spark in self.sparks:
            spark.step()

        # Create fireworks
        self.timer += t_delta_ms / 1000
        if self.timer > self.interval:
            self.timer -= self.interval

            v = np.random.choice(self.d.vertices)
            starts = [e.leds[0] if (e.v0 is v) else e.leds[-1] for e in v.edges]
            color = self.hls_to_rgb(np.random.rand(), 0.7, 1)

            spark0 = Particle(starts[0], color, 10, trail_style=self.trail_style, duration=np.random.randint(*self.duration_between))
            spark0.blocked = [starts[1], starts[2]]
            self.sparks.append(spark0)

            spark1 = Particle(starts[1], color, 10, trail_style=self.trail_style, duration=np.random.randint(*self.duration_between))
            spark1.blocked = [starts[0], starts[2]]
            self.sparks.append(spark1)

            spark2 = Particle(starts[2], color, 10, trail_style=self.trail_style, duration=np.random.randint(*self.duration_between))
            spark2.blocked = [starts[1], starts[0]]
            self.sparks.append(spark2)



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
        for edge in self.d.edges:
            for led in edge.leds[::2]:
                hue = np.sin(led.theta + self.speed * time())/2 + 0.5
                lightness = np.sin(self.speed * time())/4 + 0.25
                led.color = np.array(hls_to_rgb(hue, lightness, 1.0)) * 255
            for led in edge.leds[1::2]:
                hue = np.sin(led.theta + self.speed * time())/2 + 0.5
                lightness = np.sin(self.speed * time() + np.pi)/4 + 0.25
                led.color = np.array(hls_to_rgb(hue, lightness, 1.0)) * 255


class Hamiltonian(Animation):

    def __init__(self, dodecahedron):
        super().__init__(dodecahedron)
        self.path = self.d.find_hamiltonian(np.random.choice(self.d.vertices))

    def step(self, t_delta_ms=0):
        # Dim all previous colors
        for led in self.d.leds:
            led.set_color(led.color * 0.4)
        # Light up every n-th LED with time dependent rainbow
        offset = time() / 4
        for i in range(int(time()*15)%5, len(self.path), 5):
            self.path[i].set_color(self.hls_to_rgb((i/len(self.path) + offset)%1))


class FlashOpposingEdges(Animation):

    def __init__(self, dodecahedron, interval=.25):
        super().__init__(dodecahedron)
        self.interval = interval
        self.timer = 0

        centers = [(e.v0.pos + e.v1.pos)/2 for e in self.d.edges]
        self.pairs = []
        for i in range(len(centers)):
            j = np.argmin([np.linalg.norm(centers[i] + centers[j]) for j in range(len(centers))])
            self.pairs.append((self.d.edges[i], self.d.edges[j]))

    def step(self, t_delta_ms=0):
        # Dim existing colors
        for led in self.d.leds:
            led.set_color(led.color * 0.9)

        # Every <interval> seconds, flash new pair
        self.timer += t_delta_ms / 1000
        if self.timer > self.interval:
            self.timer -= self.interval
            print(self.timer)

            eligible = [p for p in self.pairs if p[0].leds[0].color.mean() < 2.5]
            pair = eligible[np.random.choice(len(eligible))]
            color = self.random_rgb()
            pair[0].fill(color)
            pair[1].fill(color)




class StreamFromCorner(Animation):
    pass


class RotatingTiles(Animation):
    pass




ANIMATION_CYCLE = [
    Lightning,
    CornerFireworks,
    FlashOpposingEdges,
    PulsingCorners,
    RainbowWorms,
    Hamiltonian,
    LightUpFaces,
    RollingHue,
    TrailingSparks,
    Glitter,
]
