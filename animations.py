import numpy as np
from colorsys import hls_to_rgb
from time import time
from collections import deque
from itertools import chain

from colors import *


# Base class, maybe unneccessary
class Animation():

    def __init__(self, dodecahedron, backlight=(0.,0.,0.)):
        self.d = dodecahedron

    def initialize(self):
        self.d.colors.fill(0)

    def clean_up(self):
        pass

    def hermite(self, t):
        return 3*t*t - 2*t*t*t


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
                led.color = led.color * self.trail_factor
            elif self.trail_style == 'linear':
                led.color = np.maximum(0, led.color - self.color_decrement)
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
                        self.color = clamp(self.color - self.color_decrement/2)
                        if self.color.sum() == 0:
                            self.state = self.DEAD
                if self.state == self.DEAD:
                    if np.sum([led.color for led in self.trail]) < 1:
                        for led in self.trail:
                            led.turn_off()
                        self.state = self.GONE


        # Move particle further along LED graph
        if self.state in [self.ALIVE, self.DYING]:
            if self.led.neighbors_downstream is None:
                next_options = [n for n in self.led.neighbors if n not in self.blocked]
            else:
                next_options = [n for n in self.led.neighbors_downstream if n not in self.blocked]

            if len(next_options) == 0:
                self.duration = 0
                self.state = self.DEAD
                next = self.led
            elif len(next_options) == 1:
                next = next_options[0]
            elif self.swarm is None:
                next = np.random.choice(next_options)
            else:
                # In swarm mode, try to move away from nearest swarm member
                d_swarm = [(np.linalg.norm(self.led.pos - p.led.pos) + 0.001*np.random.rand(), p) for p in self.swarm if p is not self]
                nearest_pos = sorted(d_swarm)[0][1].led.pos
                d_options = [(np.linalg.norm(led.pos - nearest_pos) + 0.001*np.random.rand(), led) for led in next_options]
                next = sorted(d_options)[-1][1]

            if next is not None:
                self.blocked = next_options + [self.led]
                self.led = next
                self.led.color = clamp(self.led.color + self.color)



# Various animations

class TrailingSparks(Animation):

    def __init__(self, dodecahedron, n_sparks=30, trail_style='linear'):
        super().__init__(dodecahedron)

        # Create sparks
        self.sparks = []
        for i in range(n_sparks):
            start = np.random.choice(self.d.leds)
            spark = Particle(start, random_rgb(), 10, trail_style=trail_style)
            self.sparks.append(spark)

    def step(self, t_delta_ms=0):
        for spark in self.sparks:
            spark.step()


class RainbowWorms(Animation):

    def __init__(self, dodecahedron, n_worms=8):
        super().__init__(dodecahedron)
        self.worm_length = 5 * self.d.leds_per_edge
        self.rainbow = [hls_to_rgb(h) for h in np.linspace(0, 1, self.worm_length)]

        # Create worms
        self.worms = []
        for i in range(n_worms):
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
            activated.color = random_rgb()
            self.active.append(activated)

            self.d.colors.fill(0)
            for face in self.active:
                for led in face.leds:
                    led.color = np.minimum(255, led.color + face.color)

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
        self.d.colors.fill(0)

        for c in self.corners:
            factor = (1 + np.sin(c.speed * time() + c.phase))/2
            if factor < 2/self.d.leds_per_edge:
                c.reroll()
            for leds, extent in zip(c.leds, c.extents):
                current_extent = int(extent * factor)
                for i in range(current_extent):
                    l = 0.25 + (i/current_extent)/2
                    s = 0.25 + 0.75 * (i/current_extent)
                    leds[i].color = hls_to_rgb(c.hue, l, s)


class Lightning(Animation):

    class Strike():
        def __init__(self, leds, sequence, color):
            self.state = 0
            self.leds = leds
            self.sequence = sequence
            self.color = color

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
            origin = self.d.leds[np.random.randint(len(self.d.leds))]
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

            color = hls_to_rgb(np.random.rand(),.9,1)
            self.strikes.append(self.Strike(leds, sequence, color))

        # Dim previous lights
        self.d.colors *= self.afterglow

        # Display lightning
        for strike in self.strikes:
            for led in strike.leds:
                led.color = strike.color * strike.sequence[strike.state]
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
            color = hls_to_rgb(np.random.rand(), 0.7, 1)

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

    def __init__(self, dodecahedron, n_per_step=15, afterglow=0.8, colored=True):
        super().__init__(dodecahedron)
        self.n_per_step = n_per_step
        self.afterglow = afterglow
        self.color = None if colored else np.array([255,255,255])

    def step(self, t_delta_ms=0):
        # Dim previous lights
        self.d.colors *= self.afterglow

        # Spawn new glitter lights
        for i in range(self.n_per_step):
            if self.color is None:
                color = hls_to_rgb(np.random.rand(), 0.8, 1)
            else:
                color = self.color
            led = self.d.leds[np.random.randint(len(self.d.leds))]
            led.color = clamp(led.color + color)


class RollingHue(Animation):

    def __init__(self, dodecahedron, speed=2.0):
        super().__init__(dodecahedron)
        self.speed = speed
        self.hls = np.zeros((len(self.d.leds), 3))
        self.hls[:,2] = 1

    def step(self, t_delta_ms=0):
        phase = self.speed * time()
        thetas = np.array([led.theta for led in self.d.leds])

        self.hls[:,0] = np.sin(thetas + phase)/2 + 0.5
        self.hls[0::2,1] = np.sin(phase)/4 + 0.25
        self.hls[1::2,1] = np.sin(phase + np.pi)/4 + 0.25
        np.copyto(self.d.colors, hls_to_rgb_array(self.hls))


class Hamiltonian(Animation):

    def __init__(self, dodecahedron):
        super().__init__(dodecahedron)
        self.path = self.d.find_hamiltonian(np.random.choice(self.d.vertices))

    def step(self, t_delta_ms=0):
        # Dim all previous colors
        self.d.colors *= 0.4
        # Light up every n-th LED with time dependent rainbow
        offset = time() / 4
        for i in range(int(time()*15)%5, len(self.path), 5):
            self.path[i].color = hls_to_rgb((i/len(self.path) + offset)%1)


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
        self.d.colors *= 0.9

        # Every <interval> seconds, flash new pair
        self.timer += t_delta_ms / 1000
        if self.timer > self.interval:
            self.timer -= self.interval

            eligible = [p for p in self.pairs if p[0].leds[0].color.mean() < 2.5]
            pair = eligible[np.random.choice(len(eligible))]
            color = random_rgb()
            pair[0].fill(color)
            pair[1].fill(color)


class RotatingTiles(Animation):

    class Tile():

        def __init__(self, face):
            self.face = face
            self.reset()

        def reset(self):
            self.progress = 0
            self.timer = 0
            self.duration = 0.5 + 2.5 * np.random.rand()
            self.cooldown = 10
            self.reversed = np.random.rand() < 1/2

    def __init__(self, dodecahedron):
        super().__init__(dodecahedron)

        self.inactive = [self.Tile(f) for f in self.d.faces]
        np.random.shuffle(self.inactive)
        self.active = [self.inactive.pop()]

    def initialize(self):
        super().initialize()

        # Fill each edge with its own color and pattern
        for edge in self.d.edges:
            hue_start = np.random.rand()
            hue_range = 0.3 * np.random.rand()

            for i in range(2,self.d.leds_per_edge//2):
                x = i - self.d.leds_per_edge//2 + 1
                stretch = 0.5 + 5.0 * np.random.rand()
                lightness = (1 + np.cos(stretch * x))/2
                lightness = 0.66 * lightness**2
                color = random_hue(hue_start, hue_start+hue_range, lightness, 1)
                edge.leds[i].set_color(color)
                edge.leds[-(i+1)].set_color(color)

            edge.leds[self.d.leds_per_edge//2].color = hls_to_rgb(hue_start + hue_range/2)

    def step(self, t_delta_ms=0):

        # Rotate active tiles
        for tile in self.active:
            # Only update at the speed needed for given duration
            tile.timer += t_delta_ms / 1000
            if (tile.progress/self.d.leds_per_edge) < (tile.timer/tile.duration):
                # Rotate by one LED in set direction
                leds = tile.face.leds
                if not tile.reversed:
                    first = np.array(leds[0].color)
                    for i in range(len(leds) - 1):
                        leds[i].color = leds[i+1].color
                    leds[-1].color = first
                else:
                    first = np.array(leds[-1].color)
                    for i in range(len(leds)-1, 0, -1):
                        leds[i].color = leds[i-1].color
                    leds[0].color = first
                tile.progress += 1

        # Update inactive tiles
        for tile in self.inactive:
            tile.cooldown -= 1

        # Find an inactive tile that has no currently active neighbors and activate it
        active_faces = [t.face for t in self.active]
        activate = None
        for i, tile in enumerate(self.inactive):
            if all(n not in active_faces for n in tile.face.neighbors) and tile.cooldown <= 0:
                activate = i
                break

        if activate is not None:
            self.active.append(self.inactive.pop(activate))

        # Update active and inactive lists
        active_updated = []
        for tile in self.active:
            if tile.progress >= self.d.leds_per_edge:
                tile.reset()
                self.inactive.append(tile)
            else:
                active_updated.append(tile)
        self.active = active_updated
        np.random.shuffle(self.inactive)


class WanderingLanterns(Animation):

    class Lantern():

        def __init__(self, color, speed):
            self.color = color
            self.speed = speed
            self.theta = np.random.rand() * 2*np.pi
            self.phi =  np.random.rand() * 2*np.pi
            self.v_theta = speed * np.random.randn()
            self.v_phi = speed * np.random.randn()

        def cartesian(self, r=np.sqrt(3)):
            pos = np.zeros(3)
            pos[0] = r * np.sin(self.theta) * np.cos(self.phi)
            pos[1] = r * np.sin(self.theta) * np.sin(self.phi)
            pos[2] = r * np.cos(self.theta)
            return pos

    def __init__(self, dodecahedron, n=10, speed=0.05):
        super().__init__(dodecahedron)
        self.lanterns = [self.Lantern(random_hue(), speed) for i in range(n)]
        self.lantern_colors = np.stack([l.color for l in self.lanterns])
        self.scale = 3
        self.spread = .3
        self.cutoff = 1.5 * np.sqrt(self.spread)
        self.pos_leds = np.vstack([l.pos for l in self.d.leds])

    def step(self, t_delta_ms=0):
        # Move lanterns
        for lantern in self.lanterns:
            lantern.v_theta = 0.75 * (lantern.v_theta + lantern.speed * np.random.randn())
            lantern.v_phi = 0.75 * (lantern.v_phi + lantern.speed * np.random.randn())
            lantern.theta = (lantern.theta + lantern.v_theta) % (2*np.pi)
            lantern.phi = (lantern.phi + lantern.v_phi) % (2*np.pi)

        # Vectorized distance matrix for efficiency
        pos_lanterns = np.vstack([l.cartesian(self.d.radius) for l in self.lanterns])
        dists = np.sum((self.pos_leds[:,None,:] - pos_lanterns[None,:,:])**2, axis=2)**0.5

        # Add up light contributions from all lanterns via radial basis function
        rbf = self.scale * np.power(np.e, -dists/self.spread)
        colors = np.sum(rbf[:,:,None] * self.lantern_colors[None,:,:], axis=1)
        np.copyto(self.d.colors, clamp(colors))


class StreamFromCorner(Animation):

    def __init__(self, dodecahedron, interval=.25, duration=10, trail_style='exponential'):
        super().__init__(dodecahedron)
        self.interval = interval
        self.timer = 0
        self.duration = duration
        self.trail_style = trail_style
        self.sparks = []
        self.pos_leds = np.vstack([l.pos for l in self.d.leds])

    def initialize(self):
        self.d.colors.fill(0)

        # Choose new corner to stream particles from
        self.corner = np.random.choice(self.d.vertices)
        self.starts = [e.leds[0] if (e.v0 is self.corner) else e.leds[-1] for e in self.corner.edges]
        self.hue_low = np.random.rand()
        self.hue_high = self.hue_low + 0.2 * np.random.rand()
        self.elapsed = 0

        # Determine downstream neighbors for each LED, i.e. neighbors that are further away from source
        dists = np.linalg.norm(self.pos_leds - self.corner.pos, axis=1)
        # Block access to edges where both ends are equally far from source
        blocked = list(chain(*[[e.leds[0], e.leds[-1]] for e in self.d.edges if np.abs(dists[e.leds[0].i] - dists[e.leds[-1].i]) < 0.001]))
        for led in self.d.leds:
            led.neighbors_downstream = [n for n in led.neighbors if dists[n.i] > dists[led.i] + 0.001 and n not in blocked]

    def clean_up(self):
        for led in self.d.leds:
            led.neighbors_downstream = None

    def step(self, t_delta_ms=0):
        # Update existing sparks
        self.sparks = [s for s in self.sparks if s.state != Particle.GONE]
        for spark in self.sparks:
            spark.step()

        # Stream new sparks for <duration>, then let them die and switch corner
        self.elapsed += t_delta_ms / 1000
        if self.elapsed < self.duration:
            self.timer += t_delta_ms / 1000
            if self.timer > self.interval:
                self.timer -= self.interval

                color = random_hue(self.hue_low, self.hue_high, 0.6, 1)
                self.starts = self.starts[1:] + self.starts[:1]
                spark = Particle(self.starts[0], color, 10, trail_style=self.trail_style)
                spark.blocked = [self.starts[1], self.starts[2]]
                self.sparks.append(spark)
        else:
            if len(self.sparks) == 0:
                self.initialize()


ANIMATION_CYCLE = [
    StreamFromCorner,
    WanderingLanterns,
    RotatingTiles,
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
