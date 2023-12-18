import numpy as np
from itertools import chain

from animations import *


# Graph elemenets

class Vertex():

    def __init__(self, pos, neighbors=None, edges=None):
        self.pos = pos
        if neighbors is None:
            self.neighbors = []
        else:
            self.neighbors = list(neighbors)
        if edges is None:
            self.edges = []
        else:
            self.edges = list(edges)

        # Polar coordinates
        x,y,z = pos
        xy = x*x + y*y
        self.theta = np.arctan2(z, np.sqrt(xy))
        self.phi = np.arctan2(y,x)

class Edge():

    def __init__(self, v0, v1, leds_per_edge):
        # Neighborhood relations
        self.v0 = v0
        self.v1 = v1
        v0.neighbors.append(v1)
        v1.neighbors.append(v0)
        v0.edges.append(self)
        v1.edges.append(self)
        self.faces = []

        # Create the LED objects along this edge
        self.leds = [LED((1-t)*v0.pos + t*v1.pos) for t in np.linspace(0, 1, leds_per_edge+2)[1:-1]]

    def flip(self):
        self.v0, self.v1 = self.v1, self.v0
        self.leds = self.leds[::-1]

    def fill(self, color):
        for led in self.leds:
            led.set_color(color)

    def turn_off(self):
        for led in self.leds:
            led.turn_off()

class Face():

    def __init__(self, normal, all_vertices, all_edges):
        # Find the five vertices by proximity to face normal
        self.vertices = [v for v in all_vertices if np.linalg.norm(v.pos - normal) < 2]

        # Find the according edges and orient them to get a continuous list of LEDs
        unsorted_edges = set([e for e in all_edges if e.v0 in self.vertices and e.v1 in self.vertices])
        self.edges = [unsorted_edges.pop()]
        self.leds = list(self.edges[0].leds)
        while len(unsorted_edges) > 0:
            next_edge = [e for e in unsorted_edges if self.edges[-1].v1 in (e.v0, e.v1)][0]
            if self.edges[-1].v1 == next_edge.v1:
                next_edge.flip()
            unsorted_edges.remove(next_edge)
            self.edges.append(next_edge)
            self.leds.extend(next_edge.leds)

        # Neighborhood stuff
        self.neighbors = []
        for e in self.edges:
            self.neighbors.extend(e.faces)
            e.faces.append(self)
        for neighbor in self.neighbors:
            if not self in neighbor.neighbors:
                neighbor.neighbors.append(self)

        # Initial color is off
        self.color = np.zeros(3)

    def fill(self, color):
        for led in self.leds:
            led.set_color(color)

    def turn_off(self):
        for led in self.leds:
            led.turn_off()

class LED():

    def __init__(self, pos, neighbors=None):
        self.pos = pos
        if neighbors is None:
            self.neighbors = []
        else:
            self.neighbors = list(neighbors)
        self.i = None
        self.colors = None

        # Polar coordinates
        x,y,z = pos
        xy = x*x + y*y
        self.theta = np.arctan2(z, np.sqrt(xy))
        self.phi = np.arctan2(y,x)

    def get_color(self):
        return self.colors[self.i,:]

    def set_color(self, color):
        self.colors[self.i,:] = color

    color = property(get_color, set_color)

    def turn_off(self):
        self.colors[self.i,:].fill(0)


# Main class

class Dodecahedron():

    def __init__(self, leds_per_edge):
        self.leds_per_edge = leds_per_edge

        # Dodecahedron coordinates
        r = (1 + np.sqrt(5)) / 2 # golden ratio
        coords = np.array([
            (1, 1, 1),   (1, 1, -1),   (1, -1, 1),   (1, -1, -1),
            (-1, 1, 1),  (-1, 1, -1),  (-1, -1, 1),  (-1, -1, -1),
            (0, r, 1/r), (0, r, -1/r), (0, -r, 1/r), (0, -r, -1/r),
            (1/r, 0, r), (1/r, 0, -r), (-1/r, 0, r), (-1/r, 0, -r),
            (r, 1/r, 0), (r, -1/r, 0), (-r, 1/r, 0), (-r, -1/r, 0)
        ])
        # Icosahedron coordinates (correspond to dodecahedron face normals)
        face_normals = np.array([
            (0, 1, r), (0, 1, -r), (0, -1, r), (0, -1, -r),
            (1, r, 0), (1, -r, 0), (-1, r, 0), (-1, -r, 0),
            (r, 0, 1), (r, 0, -1), (-r, 0, 1), (-r, 0, -1)
        ])
        self.radius = np.sqrt(3)

        # Construct dodecahedron graph
        self.vertices = [Vertex(c) for c in coords]
        self.edges = []
        for i in range(len(coords)):
            for j in range(i):
                if 0 < np.linalg.norm(coords[i] - coords[j]) <= 2/r + 0.01:
                    self.edges.append(Edge(self.vertices[i], self.vertices[j], self.leds_per_edge))
        for v in self.vertices:
            e0, e1, e2 = v.edges
            if v is e0.v0: e0.neighbors0 = [e1,e2]
            if v is e0.v1: e0.neighbors1 = [e1,e2]
            if v is e1.v0: e1.neighbors0 = [e0,e2]
            if v is e1.v1: e1.neighbors1 = [e0,e2]
            if v is e2.v0: e2.neighbors0 = [e0,e1]
            if v is e2.v1: e2.neighbors1 = [e0,e1]

        # Connect LEDs with underlying color array
        self.leds = list(chain(*[e.leds for e in self.edges]))
        self.colors = np.zeros((len(self.leds), 3))
        for i, led in enumerate(self.leds):
            led.i = i
            led.colors = self.colors

        # Construct separate LED neighborhood graph
        for i in range(len(self.leds)):
            for j in range(i):
                if 0 < np.linalg.norm(self.leds[i].pos - self.leds[j].pos) <= 1.999 * (2/r) / (leds_per_edge + 1):
                    self.leds[i].neighbors.append(self.leds[j])
                    self.leds[j].neighbors.append(self.leds[i])

        # Set up faces in their own structure
        self.faces = [Face(fn, self.vertices, self.edges) for fn in face_normals]

        # Get animations going
        self.animation = None
        self.animation_index = -1
        self.next_animation()

    def find_hamiltonian(self, start, current=None, visited=None, current_path=None):
        # Initialize recursion variables
        if current is None:
            current = start
            visited = []
            current_path = []

        visited.append(current)

        for edge in np.random.permutation(current.edges):
            # Orient the edge
            if edge.v0 is not current:
                edge.flip()
            # Check for success, in that case return LED list
            if (edge.v1 is start) and (len(visited) == len(self.vertices)):
                self.hamiltonian = current_path + edge.leds
                return current_path + edge.leds
            # If edge leads to new vertex, continue recursion
            if edge.v1 not in visited:
                found = self.find_hamiltonian(start, edge.v1, list(visited), current_path + edge.leds)
                # If success further down the line, pass upwards
                if found is not None:
                    return found

        # If no cycle found, return None
        return None

    def turn_off(self):
        for led in self.leds:
            led.turn_off()

    def get_leds(self):
        return np.stack([led.pos for led in self.leds])

    def get_colors(self):
        return np.stack([led.color for led in self.leds])

    def next_animation(self, prev=False):
        if self.animation is not None:
            self.animation.clean_up()
        self.animation_index = (self.animation_index + (-1 if prev else 1)) % len(ANIMATION_CYCLE)
        self.animation = ANIMATION_CYCLE[self.animation_index](self)
        self.animation.initialize()




if __name__ == '__main__':
    d = Dodecahedron(17)
