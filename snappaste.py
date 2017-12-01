"""
use snappaste algorithm to merge two meshes
https://www.cs.tau.ac.il/~dcor/articles/2006/SnapPaste.pdf
"""
import argparse
import math
from collections import Counter

import numpy as np
from plyfile import PlyData
from trimesh import Trimesh

from updateable_priority_queue import UpdateablePriorityQueue


class Mesh:
    def __init__(self, plydata):
        vertices = plydata["vertex"].data.view((np.float32, 6))
        self.positions = vertices[:, :3]
        self.normals = vertices[:, 3:]

        self.faces = [face for (face,) in plydata["face"].data]

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

    def find_boundary_loop(self):
        """assumes there is only one boundary loop"""
        edges = Counter()
        for face in self.faces:
            prev = face[len(face-1)]
            for vertex in face:
                edge = Edge(vertex, prev)
                edges[edge] += 1
                prev = vertex
        loop = set()
        for (edge, count) in edges.items():
            if count == 1:
                loop.add(edge[0])
                loop.add(edge[1])
        return loop

class Edge(tuple):
    def __new__(self, vertex1, vertex2):
        return tuple.__new__(self, (min(vertex1, vertex2), max(vertex1, vertex2)))

def get_closest_vertices_in_other_mesh(mesh1, vertices, mesh2):
    output = set()
    for vertex in vertices:
        v1 = mesh1.positions[vertex]
        closest = (math.inf,)
        for (i, v2) in enumerate(mesh2.positions):
            dist = np.linalg.norm(v1 - v2)
            closest = min(closest, (dist, i))
        output.add(closest[1])
    return output

def find_snapping_region(mesh, boundary_loop, closest_vertices_to_other_mesh):
    """
    returns the set of all vertices in the mesh with a geodesic distance to the boundary_loop
    less than or equal to any vertex in closest_vertices_to_other_mesh
    """
    distances = {vertex: math.inf for vertex in len(mesh.positions)}
    for vertex in boundary_loop:
        distances[vertex] = 0
    queue = UpdateablePriorityQueue(distances)

    targets = closest_vertices_to_other_mesh.copy()
    region = set()
    region_size = 0
    dist = 0

    while targets or dist < region_size * 2:
        (dist, vertex) = queue.pop()
        distances[vertex] = dist
        if targets:
            region.add(vertex)
            region_size = dist
            if vertex in targets:
                targets.remove(vertex)

        for adjacent in mesh.adjacency_list[vertex]:
            adj_dist = dist + mesh.get_vertex_distance(vertex, adjacent)
            if adj_dist < queue[adjacent]:
                queue[adjacent] = adj_dist

    return region, region_size, distances

(w1, w2, w3) = (.6, .2, .2)
def calculate_correspondence_distance(mesh1, vertex1, mesh2, vertex2):
    v1 = mesh1.positions[vertex1]
    v2 = mesh2.positions[vertex2]
    dist = w1 * np.linalg.norm(v1 - v2)

    n1 = mesh1.normals[vertex1]
    n2 = mesh2.normals[vertex2]
    dist += w2 * math.acos(np.dot(n1, n2))

    c1 = calculate_gaussian_curvature(mesh1, vertex1) # TODO: implement gaussian curvature
    c2 = calculate_gaussian_curvature(mesh2, vertex2)
    dist += w3 * abs(c1 - c2)

    return dist

def find_correspondence_points(from_mesh, from_snapping_region, to_mesh, to_snapping_region):
    correspondence = {}
    for from_vertex in from_snapping_region:
        min_dist = (math.inf,)
        for to_vertex in to_snapping_region:
            dist = calculate_correspondence_distance(from_mesh, from_vertex, to_mesh, to_vertex)
            min_dist = min(dist, min_dist)
        correspondence[from_vertex] = min_dist[1]
    return correspondence

def find_supporting_neighborhood(mesh, vertex, snapping_region_size, iteration, distance_to_boundary, elasticity):
    min_dist = snapping_region_size * math.exp(-math.pow(iteration * elasticity / distance_to_boundary, 2))

    # find all vertices within min_dist geodesic distance
    #TODO: cache distance between points?
    init_distances = {vertex: math.inf for vertex in len(mesh.positions)}
    init_distances[vertex] = 0
    queue = UpdateablePriorityQueue(init_distances)

    neighborhood = set()

    while True:
        (dist, vertex) = queue.pop()
        if dist > min_dist:
            break
        neighborhood.add(vertex)

        for adjacent in mesh.adjacency_list[vertex]:
            adj_dist = dist + mesh.get_vertex_distance(vertex, adjacent)
            if adj_dist < queue[adjacent]:
                queue[adjacent] = adj_dist

    return neighborhood

def calculate_scaling(local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations):
    # scale oriented bounding box to be same size
    # use trimesh for min OBBs
    local_trimesh = Trimesh([local_positions[v] for v in local_neighborhood])
    local_extents = local_trimesh.bounding_box_oriented.primitive.extents.copy()
    local_extents.sort()
    corresponding_trimesh = Trimesh([corresponding_positions.values()])
    corresponding_extents = corresponding_trimesh.bounding_box_oriented.primitive.extents.copy()
    corresponding_extents.sort()
    scaling = corresponding_extents / local_extents
    scaling *= iteration / max_iterations
    return np.matrix(np.diag(np.append(scaling, 1))) # matlib's diag() still makes arrays

def calculate_translation(faces, local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations):
    # translate center of mass to origin, then to center of mass of corresponding
    # use triangles in the local neighborhood for both centers of mass
    local_center = np.zeros(3)
    local_weight = 0
    corresponding_center = np.zeros(3)
    corresponding_weight = 0
    for face in faces:
        if len(face) != 3:
            raise ValueError("mesh is not a triangle mesh")
        is_in_local_neighborhood = True
        for v in face:
            if v not in local_neighborhood:
                is_in_local_neighborhood = False
        if not is_in_local_neighborhood:
            continue

        pos = [local_positions[vertex] for vertex in face]

        local_face_center = sum(pos) / 3
        local_face_weight = np.linalg.norm(np.cross(pos[1] - pos[0], pos[2] - pos[0]))

        local_center += local_face_center * local_face_weight
        local_weight += local_face_weight

        pos = [corresponding_positions[vertex] for vertex in face]

        corresponding_face_center = sum(pos) / 3
        corresponding_face_weight = np.linalg.norm(np.cross(pos[1] - pos[0], pos[2] - pos[0]))

        corresponding_center += corresponding_face_center * corresponding_face_weight
        corresponding_weight += corresponding_face_weight
    local_center /= local_weight
    corresponding_center /= corresponding_weight
    # translation = corresponding_center - local_center
    T1 = np.matlib.eye(4)
    T1[:-1,-1] = -local_center * iteration / max_iterations
    T2 = np.matlib.eye(4)
    T2[:-1,-1] = corresponding_center * iteration / max_iterations
    return T1, T2, local_center, corresponding_center 

def calculate_rotation(local_neighborhood, local_positions, local_center, corresponding_positions, corresponding_center, iteration, max_iterations):
    # rotate to minimize distance between corresponding points (using SVD)
    # matrix = np.zeros((3,3))
    # for vertex in local_neighborhood:
    #     loc_pos = local_positions[vertex]
    #     cor_pos = corresponding_positions[vertex]
    #     # TODO: does this need to be scaled?
    #     matrix += np.outer(loc_pos - local_center, cor_pos - corresponding_center)
    # u, s, v = np.linalg.svd(matrix)
    # rotation_matrix = np.dot(v, u.T)
    # # handle special reflection case
    # if np.linalg.det(rotation_matrix) < 0:
    #     rotation_matrix[:,2] *= -1
    # R = np.matlib.eye(4)
    # R[:-1,:-1] = rotation_matrix
    # return R
    matrix = np.zeros((4,4))
    for vertex in local_neighborhood:
        loc_pos = local_positions[vertex] - local_center
        cor_pos = corresponding_positions[vertex] - corresponding_center
        # TODO: does this need to be scaled?
        loc_mat = np.array([
            [0, loc_pos[0], loc_pos[1], loc_pos[2]],
            [-loc_pos[0], 0, -loc_pos[2], loc_pos[1]],
            [-loc_pos[1], loc_pos[2], 0, -loc_pos[0]],
            [-loc_pos[2], -loc_pos[1], loc_pos[0], 0]
        ])
        cor_mat = np.array([
            [0, -cor_pos[0], -cor_pos[1], -cor_pos[2]],
            [cor_pos[0], 0, -cor_pos[2], cor_pos[1]],
            [cor_pos[1], cor_pos[2], 0, -cor_pos[0]],
            [cor_pos[2], -cor_pos[1], -cor_pos[0], 0]
        ])
        matrix += np.dot(loc_mat, cor_mat)
    vals, vecs, = np.linalg.eig(matrix)
    rotation_quaternion = vecs[:,np.argmax(vals)]

    # convert quaternion to transformation matrix
    q = rotation_quaternion[:3]
    r = rotation_quaternion[3] * iteration / max_iterations
    Q = np.matrix([
        [0, -q[2], q[1]],
        [q[2], 0, -q[0]],
        [-q[1], q[0], 0]
    ])

    rotation = (r*r - np.inner(q, q)) * np.matlib.eye(3) + 2 * np.outer(q, q) + 2 * r * Q
    R = np.matlib.eye(4)
    R[:-1,:-1] = rotation
    return R

def calculate_transforms(from_mesh, distances_to_boundary, snapping_region_size, to_mesh, correspondence_points, iteration, max_iterations, elasticity):
    transforms = {}
    for vertex in range(len(from_mesh.positions)):
        # find local neighborhood for each vertex in the mesh
        dist = distances_to_boundary[vertex]
        if dist == math.inf:
            # TODO make sure this doesn't skip anything relevant?
            transforms[vertex] = np.matlib.eye(4)
            continue

        local_neighborhood = find_supporting_neighborhood(from_mesh, vertex, snapping_region_size, iteration, dist, elasticity)

        local_positions = from_mesh.positions

        # assume correspondence point is self if not in correspondence_points
        corresponding_positions = {}
        is_corresponding_same_as_local = True
        for v in local_neighborhood:
            if v in correspondence_points:
                pos = to_mesh.positions[correspondence_points[v]]
                is_corresponding_same_as_local = False
            else:
                pos = from_mesh.positions[v]
            corresponding_positions[v] = pos
        if is_corresponding_same_as_local:
            transforms[vertex] = np.matlib.eye(4)
            continue

        # compute transform for each vertex in the mesh
        S = calculate_scaling(local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations)

        T1, T2, local_center, corresponding_center = calculate_translation(from_mesh.faces, local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations)

        R = calculate_rotation(local_neighborhood, local_positions, local_center, corresponding_positions, corresponding_center, iteration, max_iterations)

        # translate to origin, scale, rotate, translate to corresponding
        transforms[vertex] = T2 * R * S * T1
    return transforms

def run_merge_iteration(mesh1, snapping_region1, snapping_region_size1, distances_to_boundary1, mesh2, snapping_region2, iteration, iterations, elasticity):
    # find correspondence points for each vertex in each snapping region
    correspondence_points = find_correspondence_points(mesh1, snapping_region1, mesh2, snapping_region2)

    transforms = calculate_transforms(mesh1, distances_to_boundary1, snapping_region_size1, mesh2, correspondence_points, iteration, iterations, elasticity)

    for vertex in range(len(mesh1.positions)):
        # apply transform
        pos = np.append(mesh1.positions[vertex], 1)
        mesh1.positions[vertex] = (transforms[vertex] * pos)[:-1]

def merge(mesh1, mesh2):
    # find boundary loops, and the vertices on the other mesh closest to them
    loop1 = mesh1.find_boundary_loop()
    closest_vertices_to_loop1_in_mesh2 = get_closest_vertices_in_other_mesh(mesh1, loop1, mesh2)

    loop2 = mesh2.find_boundary_loop()
    closest_vertices_to_loop2_in_mesh1 = get_closest_vertices_in_other_mesh(mesh2, loop2, mesh1)

    # find snapping region: compute geodesic distance of each point to boundary loop
    snapping_region1, snapping_region_size1, distances_to_boundary1 =\
            find_snapping_region(mesh1, loop1, closest_vertices_to_loop2_in_mesh1)
    snapping_region2, snapping_region_size2, distances_to_boundary2 =\
            find_snapping_region(mesh2, loop2, closest_vertices_to_loop1_in_mesh2)
    #TODO should probably stop if snapping region is too big or small

    for i in range(iterations):
        if i % 2 == 0:
            run_merge_iteration(mesh1, snapping_region1, snapping_region_size1, distances_to_boundary1, mesh2, snapping_region2, i, iterations, elasticity)
        else:
            run_merge_iteration(mesh2, snapping_region2, snapping_region_size2, distances_to_boundary2, mesh1, snapping_region1, i, iterations, elasticity)

        # TODO retriangulate


    raise NotImplementedError()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh1", type=str)
    parser.add_argument("mesh2", type=str)
    parser.add_argument("output", type=str)
    # TODO offset?
    args = parser.parse_args()

    mesh1 = Mesh(PlyData.read(args["mesh1"]))
    mesh2 = Mesh(PlyData.read(args["mesh2"]))
    merge(mesh1, mesh2)

if __name__ == "__main__":
    main()
