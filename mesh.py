import numpy as np
from collections import Counter

class Mesh:
    def __init__(self, plydata=None):
        if plydata is None:
            self.positions = [] # positions/normals NOT np arrays here
            self.normals = []
            self.faces = []
            self.adjacency_list = {}
            self.edge_lengths = {}
        else:
            vertices = plydata["vertex"].data.view((np.float32, 3))
            # vertices = plydata["vertex"].data.view((np.float32, 6))
            self.positions = vertices[:, :3]
            # self.normals = vertices[:, 3:]
            # self.normals = np.zeros_like(self.positions)

            # each face assumed to have 3 verts
            self.faces = [face for (face,) in plydata["face"].data]

            self.update_normals()

            self.adjacency_list = self._create_adjacency_list()

            self.edge_lengths = {}

    def _create_adjacency_list(self):
        adj = {}
        for face in self.faces:
            for v1 in face:
                if v1 not in adj:
                    adj[v1] = set()
                for v2 in face:
                    if v2 != v1:
                        adj[v1].add(v2)
        return adj

    def reset_edge_lengths(self):
        self.edge_lengths = {}

    def get_vertex_distance(self, vertex1, vertex2=None):
        """pass in 2 adjacent vertices, or an edge"""
        if vertex2 is None:
            edge = vertex1
        else:
            edge = Edge(vertex1, vertex2)

        if edge not in self.edge_lengths:
            v1 = self.positions[edge[0]]
            v2 = self.positions[edge[1]]
            self.edge_lengths[edge] = np.linalg.norm(v1 - v2)
        return self.edge_lengths[edge]

    def find_boundary_vertices(self):
        """assumes there is only one boundary loop"""
        edges = Counter()
        for face in self.faces:
            prev = face[len(face)-1]
            for vertex in face:
                edge = Edge(vertex, prev)
                edges[edge] += 1
                prev = vertex

        vertices = set()
        for (edge, count) in edges.items():
            if count == 1:
                vertices.add(edge[0])
                vertices.add(edge[1])
        return vertices

    def update_normals(self):
        self.normals = np.zeros_like(self.positions)
        for face in self.faces:
            v1, v2, v3 = [self.positions[v] for v in face]
            normal = np.cross(v2 - v1, v3 - v1)
            for vertex in face:
                self.normals[vertex] += normal
        self.normals /= np.linalg.norm(self.normals, axis=-1).reshape(-1, 1)

class Edge(tuple):
    def __new__(self, vertex1, vertex2):
        return tuple.__new__(self, (min(vertex1, vertex2), max(vertex1, vertex2)))

class Triangle(tuple):
    def __new__(self, vertices):
        vertices = vertices.copy()
        vertices.sort()
        return tuple.__new__(self, vertices)
