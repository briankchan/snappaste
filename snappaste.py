"""
use snappaste algorithm to merge two meshes
https://www.cs.tau.ac.il/~dcor/articles/2006/SnapPaste.pdf
"""
import argparse
import math
from collections import Counter

import numpy as np
from plyfile import PlyData, PlyElement
import trimesh
from trimesh import Trimesh

from updateable_priority_queue import UpdateablePriorityQueue


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
            self.normals = np.zeros_like(self.positions)

            # each face assumed to have 3 verts
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

class Edge(tuple):
    def __new__(self, vertex1, vertex2):
        return tuple.__new__(self, (min(vertex1, vertex2), max(vertex1, vertex2)))

def get_closest_vertices_in_other_mesh(mesh1, vertices, mesh2):
    output = set()
    for (j, vertex) in enumerate(vertices):
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
    distances = {vertex: math.inf for vertex in range(len(mesh.positions))}
    for vertex in boundary_loop:
        distances[vertex] = 0
    queue = UpdateablePriorityQueue(distances)

    targets = closest_vertices_to_other_mesh.copy()
    region = set()
    region_size = 0
    dist = 0

    #"snapping" region: p where weight(p) <= max(weight(loop points))
    #later on need arbitrary points, we handwave the 2 for now because Brian
    #regoin size is max(weight(loop points))
    # distances is distances to vertices, some p left infinity meaning d(p) >= region_size*2
    # our cost used for dijkstras is defined as distance to nearest the boundary loop
    while (targets or dist < region_size * 2) and queue:
        (dist, vertex) = queue.pop()
        distances[vertex] = dist
        if targets:
            region.add(vertex)
            region_size = dist
            if vertex in targets:
                targets.remove(vertex)

        # print(vertex, "!")
        for adjacent in mesh.adjacency_list[vertex]:
            adj_dist = dist + mesh.get_vertex_distance(vertex, adjacent)
            if adj_dist < distances[adjacent]:
                queue[adjacent] = adj_dist
                distances[adjacent] = adj_dist

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

def calculate_gaussian_curvature(a, b):
    return 0

def find_correspondence_points(from_mesh, from_snapping_region, to_mesh, to_snapping_region):
    correspondence = {}
    for from_vertex in from_snapping_region:
        min_dist = (math.inf,)
        for to_vertex in to_snapping_region:
            dist = calculate_correspondence_distance(from_mesh, from_vertex, to_mesh, to_vertex)
            min_dist = min((dist, to_vertex), min_dist)
        correspondence[from_vertex] = min_dist[1]
    return correspondence

def find_supporting_neighborhood(mesh, vertex, snapping_region_size, iteration, distance_to_boundary, elasticity):
    """
    Find region in mesh that we're snapping from that we need to account for when we're calculating the transform
    """

    min_dist = 0.0 if distance_to_boundary == 0 \
        else snapping_region_size * math.exp(-math.pow(iteration * elasticity / distance_to_boundary, 2))

    # find all vertices within min_dist geodesic distance
    #TODO: cache distance between points?
    init_distances = {vertex: math.inf for vertex in range(len(mesh.positions))}
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
    scaling_factor = iteration / max_iterations
    scaling = (local_extents + (corresponding_extents - local_extents) * scaling_factor) / local_extents
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

    if corresponding_weight == 0:
        for vertex in local_neighborhood:
            local_center += local_positions[vertex]
            local_weight += 1
            corresponding_center += corresponding_positions[vertex]
            corresponding_weight += 1

    local_center /= local_weight
    corresponding_center /= corresponding_weight
    # translation = corresponding_center - local_center
    T1 = np.matlib.eye(4)
    T1[:-1,-1] = -local_center[np.newaxis, :].T
    T2 = np.matlib.eye(4)
    t2_tweenfactor = iteration / max_iterations
    grumble_grumble = local_center * (1 - t2_tweenfactor) + corresponding_center * t2_tweenfactor
    T2[:-1,-1] = grumble_grumble[np.newaxis, :].T
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
        # find matrix that we need to find the eigenvectors of to find the quaternions
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
    """
    :param from_mesh:
    :param distances_to_boundary:
    :param snapping_region_size:
    :param to_mesh:
    :param correspondence_points: maps from a from_mesh vertex to a to_mesh vertex
    :param iteration:
    :param max_iterations:
    :param elasticity:
    :return:
    """
    transforms = {}
    for vertex in range(len(from_mesh.positions)):
        # find local neighborhood for each vertex in the mesh
        dist = distances_to_boundary[vertex]
        if dist not in distances_to_boundary:
            # TODO make sure this doesn't skip anything relevant?
            transforms[vertex] = np.matlib.eye(4)
            continue

        local_neighborhood = find_supporting_neighborhood(from_mesh, vertex, snapping_region_size, iteration, dist, elasticity)

        local_positions = from_mesh.positions

        # assume correspondence point is self if not in correspondence_points
        corresponding_positions = {} #maps from vertex id in local neighborhood to either position in to_mesh vertices or the vertex's own position
        is_corresponding_same_as_local = True #everything local ; none of the corresponding points are actually in mesh 2's snapping region
        for v in local_neighborhood:
            if v in correspondence_points: # correspondence_points keyset is snapping region point set
                pos = to_mesh.positions[correspondence_points[v]]
                is_corresponding_same_as_local = False
            else:
                pos = from_mesh.positions[v]
            corresponding_positions[v] = pos

        # if no corresponding points were in mesh 2 snapping region, then nothing to snap so identity
        if is_corresponding_same_as_local:
            transforms[vertex] = np.matlib.eye(4)
            continue

        # compute transform for each vertex in the mesh
        try:
            S = calculate_scaling(local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations)
        except:
            S = np.matlib.eye(4)

        T1, T2, local_center, corresponding_center = calculate_translation(from_mesh.faces, local_neighborhood, local_positions, corresponding_positions, iteration, max_iterations)

        R = calculate_rotation(local_neighborhood, local_positions, local_center, corresponding_positions, corresponding_center, iteration, max_iterations)

        # translate to origin, scale, rotate, translate to corresponding
        transforms[vertex] = T2 * R * S * T1
    return transforms

def run_merge_iteration(mesh1, snapping_region1, snapping_region_size1, distances_to_boundary1, mesh2,
                        snapping_region2, iteration, iterations, elasticity):
    # find correspondence points for each vertex in each snapping region
    correspondence_points = find_correspondence_points(mesh1, snapping_region1, mesh2, snapping_region2)

    # transformation for each vertex in mesh. Everything that had distnace inf should have identity matrix
    transforms = calculate_transforms(mesh1, distances_to_boundary1, snapping_region_size1, mesh2,
                                      correspondence_points, iteration, iterations, elasticity)

    for vertex in range(len(mesh1.positions)):
        # apply transform
        pos = np.append(mesh1.positions[vertex], 1)
        mesh1.positions[vertex] = np.dot(transforms[vertex], pos)[0, :-1]
    mesh1.reset_edge_lengths()

def merge(mesh1, mesh2, iterations, elasticity):
    # find unordered boundary loops, and the vertices on the other mesh closest to them
    boundary_vertices1 = mesh1.find_boundary_vertices()
    closest_vertices_to_loop1_in_mesh2 = get_closest_vertices_in_other_mesh(mesh1, boundary_vertices1, mesh2)

    boundary_vertices2 = mesh2.find_boundary_vertices()
    closest_vertices_to_loop2_in_mesh1 = get_closest_vertices_in_other_mesh(mesh2, boundary_vertices2, mesh1)

    # find snapping region: compute geodesic distance of each point to boundary loop
    snapping_region1, snapping_region_size1, distances_to_boundary1 =\
            find_snapping_region(mesh1, boundary_vertices1, closest_vertices_to_loop2_in_mesh1)
    snapping_region2, snapping_region_size2, distances_to_boundary2 =\
            find_snapping_region(mesh2, boundary_vertices2, closest_vertices_to_loop1_in_mesh2)
    #TODO should probably stop if snapping region is too big or small

    for i in range(1, iterations + 1):
        print("Enter iteration", i)
        if i % 2 == 0:
            run_merge_iteration(mesh1, snapping_region1, snapping_region_size1, distances_to_boundary1, mesh2, snapping_region2, i, iterations, elasticity)
        else:
            run_merge_iteration(mesh2, snapping_region2, snapping_region_size2, distances_to_boundary2, mesh1, snapping_region1, i, iterations, elasticity)

        # TODO retriangulate

    #raise NotImplementedError()

def save_mesh(mesh, filename):
    lists = [tuple(i) for i in mesh.positions.tolist()]
    # positions_element = PlyElement.describe(, "vertex")
    positions_element = PlyElement.describe(np.array(lists, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]), "vertex")

    faces = [(face,) for face in mesh.faces]
    faces_element = PlyElement.describe(np.array(faces, dtype=[("vertex_indices", "i4", (3,))]), "face")

    PlyData([positions_element, faces_element], text=True).write(filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh1", type=str)
    parser.add_argument("mesh2", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("iterations", type=int)
    parser.add_argument("--elasticity", type=int, default=1)
    # TODO offset?
    args = parser.parse_args()

    mesh1 = Mesh(PlyData.read(args.mesh1))
    mesh2 = Mesh(PlyData.read(args.mesh2))
    merge(mesh1, mesh2, args.iterations, args.elasticity)
    #
    import pickle
    # with open('mesh1.pickle', 'wb') as f:
    #     pickle.dump(mesh1, f)
    # with open('mesh2.pickle', 'wb') as f:
    #     pickle.dump(mesh2, f)
    # with open('mesh1.pickle', 'rb') as f1,\
    #     open('mesh2.pickle', 'rb') as f2:
    #     mesh1 = pickle.load(f1)
    #     mesh2 = pickle.load(f2)

    # IPython.embed()

    save_mesh(mesh1, "output1.ply")
    save_mesh(mesh2, "output2.ply")
    (trimesh.load("output1.ply") + trimesh.load("output2.ply")).show()


if __name__ == "__main__":
    main()
