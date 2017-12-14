
from plyfile import PlyData, PlyElement
import math
from math import acos, pi, sin, tan

import numpy as np
import scipy.optimize

from mesh import Edge, Mesh
from updateable_priority_queue import UpdateablePriorityQueue

NEIGHBORHOOD_SIZE = 5

class PointCloud():
    def __init__(self, meshes_and_regions, smoothing_factor, osculating_circle_angle_subtended):
        self.smoothing_factor = smoothing_factor
        self.osculating_circle_angle_subtended = osculating_circle_angle_subtended
        self.positions = []
        self.neighborhoods = {}
        self.guidance_field_values = {}
        for mesh, region in meshes_and_regions:
            for vertex in region:
                self.positions.append(mesh.positions[vertex])

    def find_neighborhood(self, vertex):
        """vertex can be an index from positions, or a position itself (i.e. numpy array)"""
        if isinstance(vertex, int):
            if vertex in self.neighborhoods:
                return self.neighborhoods[vertex]
            pos = self.positions[vertex]
        else:
            pos = vertex

        squared_dists = [np.dot(pos-other, pos-other) for other in self.positions]
        # find NEIGHBORHOOD_SIZE closest vertices
        neighborhood = np.argpartition(squared_dists, NEIGHBORHOOD_SIZE-1)[:NEIGHBORHOOD_SIZE]
        # get distance of farthest vertex in neighborhood
        neighborhood_radius = np.sqrt(squared_dists[neighborhood[-1]])
        output = set(neighborhood), neighborhood_radius
        if isinstance(vertex, int):
            self.neighborhoods[vertex] = output
        return output

    def find_neighborhood_and_smoothing(self, vertex):
        neighborhood, _ = self.find_neighborhood(vertex)
        position = self.positions[vertex] if isinstance(vertex, int) else vertex
        distances = []
        radii = []
        for v in neighborhood:
            distances.append(np.linalg.norm(position - self.positions[v]))
            radii.append(self.find_neighborhood(v)[1])
        # offset distances so we don't divide by 0
        distance_offset = sum(distances) / 10
        smoothing = weights = 0
        for d,r in zip(distances, radii):
            weight = 1 / (d + distance_offset)
            smoothing += r * weight
            weights += weight
        smoothing *= self.smoothing_factor / weights
        neighborhood_positions = [self.positions[v] for v in neighborhood]
        return neighborhood_positions, smoothing

    def find_closest_vertex(self, position):
        closest = (math.inf,)
        for i, vertex_position in enumerate(self.positions):
            dist = np.linalg.norm(position - vertex_position)
            if dist < closest[0]:
                closest = (dist, i)
        return closest[1]

    def calculate_principal_curvatures(self, vertex):
        neighborhood, smoothing = self.find_neighborhood_and_smoothing(vertex)
        coeffs = calculate_polynomial(self.positions[vertex], neighborhood, smoothing)

        # polynomial is centered at position, i.e. we want the curvature at (x,y) = (0,0)
        # surface defined by r(x,y) -> (x, y, z(x,y))
        # where z(x,y) = polynomial(x,y) = coeffs . vars(x,y)

        # calculate partial derivatives of r
        # vars = [1, x, y, x**2, y**2, x * y, x**3, y**3, x**2 * y, y**2 * x]
        # vars_sub_x  = coeffs . [0, 1, 0, 2x, 0, y, 3x**2, 0, 2xy, y**2]
        # vars_sub_y  = coeffs . [0, 0, 1, 0, 2y, x, 0, 3y**2, x**2, 2xy]
        # vars_sub_xx = coeffs . [0, 0, 0, 2, 0, 0, 6x, 0, 2y, 0]
        # vars_sub_yy = coeffs . [0, 0, 0, 0, 2, 0, 0, 6y, 0, 2x]
        # vars_sub_xy = coeffs . [0, 0, 0, 0, 0, 1, 0, 0, 2x, 2y]
        z_sub_x = np.dot(coeffs, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
        z_sub_y = np.dot(coeffs, np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
        z_sub_xx = np.dot(coeffs, np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 0]))
        z_sub_yy = np.dot(coeffs, np.array([0, 0, 0, 0, 2, 0, 0, 0, 0, 0]))
        z_sub_xy = np.dot(coeffs, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
        r_sub_x = [1, 0, z_sub_x]
        r_sub_y = [0, 1, z_sub_y]
        cross = np.cross(r_sub_x, r_sub_y)
        normal = cross / np.linalg.norm(cross)
        # first fundamental form coefficients (dot of first partial derivatives)
        E = np.dot(r_sub_x, r_sub_x)
        F = np.dot(r_sub_x, r_sub_y)
        G = np.dot(r_sub_y, r_sub_y)
        # second fundamental form coefficients (normal dotted with second partial derivatives)
        e = normal[2] * z_sub_xx
        f = normal[2] * z_sub_xy
        g = normal[2] * z_sub_yy

        shape_operator = np.array([[e*G-f*F, f*G-g*F], [f*E-e*F, g*E-f*F]]) / (E*G-F**2)
        principal_curvatures = np.linalg.eig(shape_operator)[0]
        return principal_curvatures

    def guidance_field(self, position):
        """guidance field for ideal triangle edge length at each position"""
        closest_vertex = self.find_closest_vertex(position)
        if closest_vertex not in self.guidance_field_values:
            print("asdf")
            curvatures = self.calculate_principal_curvatures(closest_vertex)
            value = self.osculating_circle_angle_subtended / max(np.abs(curvatures))
            self.guidance_field_values[closest_vertex] = value
        return self.guidance_field_values[closest_vertex]


def gaussian(dist, h):
    if h == 0:
        raise ValueError("Divide by 0")
    return np.exp(-dist**2 / h**2)

def calculate_projection_plane(point, neighborhood, smoothing):
    # def f(xs):
    #     t = xs[0]
    #     n = np.array(xs[1:])
    #     output = weights = 0
    #     for p in neighborhood:
    #         error = p - point - t * n
    #         weight = gaussian(np.linalg.norm(error), smoothing)
    #         output += np.dot(n, error)**2 * weight
    #         weights += weight
    #     print(t, output, weights)

    #     if weights == 0:
    #         print("inf!!")
    #         return math.inf
    #     return output / weights
    # # initial guess: t=0, n=optimum with t=0
    # def fp_at_0(xs):
    #     n = np.array(xs)
    #     # for p in neighborhood:
    #     #     pr = p-r
    #     #     mat += weight(np.linalg.norm(pr)) * np.outer(pr, pr)
    #     # scipy.linalg.solve(mat, [0, 0, 0]) # singular?
    #     output = 0
    #     for p in neighborhood:
    #         output += np.dot(n, p - point)
    #     return output
    # def unit_vector_requirement(xs):
    #     return np.linalg.norm(xs) - 1
    # n = scipy.optimize.fsolve(lambda xs: [fp_at_0(xs), unit_vector_requirement(xs), 0], [0, 0, 0])

    # output= scipy.optimize.minimize(f, [0, *n], method="Powell").x
    # print("result", f(output))
    # return output

    def f(q):
        output = weights = 0
        n = point-q
        n /= np.linalg.norm(n)
        for p in neighborhood:
            error = p - q
            weight = gaussian(np.linalg.norm(error), smoothing)
            output += np.dot(n, error)**2 * weight
            weights += weight

        if weights == 0:
            print("inf!!")
            return math.inf
        return output / weights
    output= scipy.optimize.minimize(f, [0, 0, 0], method="Powell").x
    return output

def cubic(coefficients, p):
    x,y = p
    return coefficients[0] +\
           coefficients[1] * x +\
           coefficients[2] * y +\
           coefficients[3] * x**2 +\
           coefficients[4] * y**2 +\
           coefficients[5] * x * y +\
           coefficients[6] * x**3 +\
           coefficients[8] * y**3 +\
           coefficients[7] * x**2 * y +\
           coefficients[9] * y**2 * x

def cubic_vals(p):
    x,y = p
    return np.array([1,
                     x,
                     y,
                     x**2,
                     y**2,
                     x * y,
                     x**3,
                     y**3,
                     x**2 * y,
                     y**2])

def calculate_polynomial_from_plane(point, neighborhood, plane_center, smoothing):
    # # plane_center = point + t*n
    n = point-plane_center
    n /= np.linalg.norm(n)
    ax1 = np.cross(n, n+1)
    ax1 /= np.linalg.norm(ax1)
    ax2 = np.cross(n, ax1)
    ax2 /= np.linalg.norm(ax2)
    # # def f(xs):
    # #     output = weights = 0
    # #     for p in neighborhood:
    # #         dist_to_plane = np.dot(n, p - plane_center)
    # #         proj_p1 = np.dot(ax1, p - plane_center)
    # #         proj_p2 = np.dot(ax2, p - plane_center)
    # #         proj_p = np.array([proj_p1, proj_p2])

    # #         weight = gaussian(np.linalg.norm(p - plane_center))
    # #         output += (cubic(xs, proj_p) - dist_to_plane)**2 * weight
    # #         weights += weight
    # #     return output
    # # return scipy.optimize.minimize(f, [0]*10, method="Powell").x
    # mat = np.zeros((10,10))
    # rhs = np.zeros(10)
    # for p in neighborhood:
    #     dist_to_plane = np.dot(n, p - plane_center)
    #     proj_p1 = np.dot(ax1, p - plane_center)
    #     proj_p2 = np.dot(ax2, p - plane_center)
    #     proj_p = np.array([proj_p1, proj_p2])

    #     values = cubic_vals(proj_p)
    #     weight = gaussian(np.linalg.norm(p - plane_center), smoothing)
    #     print(p - plane_center)
    #     print(np.linalg.norm(p - plane_center))
    #     print(weight, smoothing)
    #     mat += weight * np.outer(values, values)
    #     rhs += values * dist_to_plane
    # print(len(neighborhood))
    # return scipy.linalg.solve(mat, rhs)
    def f(coeffs):
        output = weights = 0
        for p in neighborhood:
            dist_to_plane = np.dot(n, p - plane_center)
            proj_p1 = np.dot(ax1, p - plane_center)
            proj_p2 = np.dot(ax2, p - plane_center)
            proj_p = np.array([proj_p1, proj_p2])
            weight = gaussian(np.linalg.norm(p - plane_center), smoothing)
            output += (cubic(coeffs, proj_p) - dist_to_plane)**2 * weight
            weights += weight

        return output / weights
    output= scipy.optimize.minimize(f, [0]*10, method="Powell").x
    return output

def calculate_polynomial(point, neighborhood, smoothing):
    """requires neighborhood of at least 3 points"""
    plane_center = calculate_projection_plane(point, neighborhood, smoothing)
    # n = np.array(n)
    # n /= np.linalg.norm(n)
    # print(t)
    output=  calculate_polynomial_from_plane(point, neighborhood, plane_center, smoothing)
    return output

def project_point(point, neighborhood, smoothing):
    """requires neighborhood of at least 3 points"""
    plane_center = calculate_projection_plane(point, neighborhood, smoothing)
    # n = np.array(n)
    # n /= np.linalg.norm(n)
    coeffs = calculate_polynomial_from_plane(point, neighborhood, plane_center, smoothing)
    # return point + t*n + coeffs[0]
    return plane_center + coeffs[0]



def field_min_in_sphere(field, center, radius):
    # def sphere(xs):
    #     return radius - np.linalg.norm(center - xs)
    # min_coords = scipy.optimize.minimize(field, center, method="COBYLA",
    #                                      constraints={"type":"ineq", "fun":sphere}).x
    # return field(min_coords)
    return .3

err_allowed = 10
min_base_angle = (60 - err_allowed) * pi / 180
max_base_angle = (60 + err_allowed) * pi / 180
def predict_vertex(edge, point_cloud, edge_other_point):
    """
    edge_other_point is the other point of the existing triangle that edge is in
    (needed to calculate direction of new vertex)
    """
    # calculate the ideal edge length
    edge_len = np.linalg.norm(edge[0] - edge[1])
    radius = edge_len * sin(2 * min_base_angle) / sin(3 * min_base_angle)
    midpoint = (edge[0] + edge[1]) / 2
    ideal_length = field_min_in_sphere(point_cloud.guidance_field, midpoint, radius)
    if ideal_length < edge_len/2:
        ideal_length = edge_len/2

    # clamp to acceptable base angle
    base_angle = acos(edge_len/2 / ideal_length)
    base_angle = np.clip(base_angle, min_base_angle, max_base_angle)

    # calculate height of triangle with clamped base angle
    height = tan(base_angle) * edge_len / 2

    # calculate direction of new vertex (in plane of prev edge's triangle)
    v1 = edge[0] - edge_other_point
    v2 = edge[1] - edge[0]
    normal_dir = v1 - np.dot(v1, v2) / np.dot(v2, v2) * v2
    normal = normal_dir / np.linalg.norm(normal_dir)

    point = midpoint + normal * height

    # project point onto MLS surface
    neighborhood, smoothing = point_cloud.find_neighborhood_and_smoothing(point)
    projected_point = project_point(point, neighborhood, smoothing)

    # calculate priority: ratio of ideal edge length to actual 
    avg_actual_length = sum(np.linalg.norm(projected_point - e) for e in edge) / 2
    # priority always >= 1; lower priority is better
    priority = ideal_length / avg_actual_length if ideal_length > avg_actual_length\
               else avg_actual_length / ideal_length

    return projected_point, priority

def find_cut_vertex(edge, mesh, prev_vertex, boundaries):
    v1 = mesh.positions[edge[0]]
    v2 = mesh.positions[edge[1]]
    for source in edge:
        for dest in mesh.adjacency_list[source]:
            if dest == prev_vertex or dest in mesh.adjacency_list[prev_vertex]: # or dest == edge[0] or edge == edge[1]
                continue
            if Edge(source,dest) not in boundaries:
                continue
            v3 = mesh.positions[dest]
            edge_lengths = [np.linalg.norm(e) for e in [v1-v2, v2-v3, v3-v1]]
            if np.any(np.less(edge_lengths, .0000000001)):
                print("ZERO", edge_lengths)
                continue
            # HACK
            if np.dot(prev_vertex - source, prev_vertex - source) < np.dot(prev_vertex - dest, prev_vertex - dest):
                print("backwards dist")
                continue
            if calculate_angle(
                np.linalg.norm(mesh.positions[source] - v3),
                np.linalg.norm(mesh.positions[prev_vertex] - mesh.positions[source]),
                np.linalg.norm(mesh.positions[prev_vertex] - v3)
            ) < 70:
                print("backwards angle")
                continue
            
            for i in range(3):
                # angle = acos((edge_lengths[i]**2 + edge_lengths[(i+1)%3]**2 - edge_lengths[(i+2)%3]**2)
                #              / (2 * edge_lengths[i] * edge_lengths[(i+1)%3]))
                angle = calculate_angle(edge_lengths[i], edge_lengths[(i+1)%3], edge_lengths[(i+2)%3])
                if angle > 70 * pi / 180:
                    # not a good triangle
                    continue
            # found a good triangle
            return dest
    return None

def calculate_angle(s1, s2, s3):
    return acos((s1**2 + s2**2 - s3**2) / (2 * s1 * s2))

def add_edge_to_boundaries(boundaries, edge, other_vertex, mesh, point_cloud):
    cut_vertex = find_cut_vertex(edge, mesh, other_vertex, boundaries)
    if cut_vertex is not None:
        next_vertex = cut_vertex
        priority = 0 # TODO should we always do cuts first?
    else:
        edge_positions = [mesh.positions[edge[0]], mesh.positions[edge[1]]]
        next_vertex, priority = predict_vertex(edge_positions, point_cloud, mesh.positions[other_vertex])
    boundaries.push(edge, (False, priority, next_vertex))

def add_edge_to_boundaries_(boundaries, edge, other_vertex_position, mesh, point_cloud):
    edge_positions = [mesh.positions[edge[0]], mesh.positions[edge[1]]]
    next_vertex, priority = predict_vertex(edge_positions, point_cloud, other_vertex_position)
    boundaries.push(edge, (False, priority, next_vertex))

def grow_triangle(mesh, edge, vertex, boundaries, point_cloud):
    print("grow")
    vertex_index = len(mesh.positions)
    mesh.positions.append(vertex)
    connect_triangle(mesh, edge, vertex_index, boundaries, point_cloud)

def connect_triangle(mesh, edge, vertex_index, boundaries, point_cloud):
    mesh.faces.append([*edge, vertex_index])
    if edge[0] == edge[1]:
        print("connect 0 edge")
    if edge[0] == vertex_index or edge[1] == vertex_index:
        print("connect new 0 edge made")

    if vertex_index not in mesh.adjacency_list:
        mesh.adjacency_list[vertex_index] = set()
    adj = mesh.adjacency_list[vertex_index]

    new_edge_1 = Edge(edge[0], vertex_index)
    if edge[0] in adj:
        if new_edge_1 in boundaries:
            boundaries.remove(new_edge_1)
        else:
            print("oh boy")
    else:
        adj.add(edge[0])
        mesh.adjacency_list[edge[0]].add(vertex_index)
        add_edge_to_boundaries(boundaries, new_edge_1, edge[1], mesh, point_cloud)

    new_edge_2 = Edge(edge[1], vertex_index)
    if edge[1] in adj:
        if new_edge_2 in boundaries:
            boundaries.remove(new_edge_2)
        else:
            print("oh boy")
    else:
        adj.add(edge[1])
        mesh.adjacency_list[edge[1]].add(vertex_index)
        add_edge_to_boundaries(boundaries, new_edge_2, edge[0], mesh, point_cloud)

def cut_ear(mesh, edge, vertex_index, boundaries, point_cloud):
    print("cut")
    mesh.faces.append([*edge, vertex_index])
    if edge[0] == edge[1]:
        print("cut ear 0 edge")
    if edge[0] == vertex_index or edge[1] == vertex_index:
        print("cut ear new 0 edge made")

    vertex_adjacencies = mesh.adjacency_list[vertex_index]
    for i, v in enumerate(edge):
        if v in vertex_adjacencies:
            # edge not new
            old_edge = Edge(vertex_index, v)
            if old_edge in boundaries:
                boundaries.remove(old_edge)
            else:
                print("disaster?")
        else:
            # edge is new
            vertex_adjacencies.add(v)
            mesh.adjacency_list[v].add(vertex_index)

            other_edge_vertex = edge[(i+1)%2]
            add_edge_to_boundaries(boundaries, Edge(v, vertex_index), other_edge_vertex, mesh, point_cloud)

def add_snapping_region_boundary(boundaries, new_mesh, mesh, snapping_region, point_cloud):
    boundary_vertices = set()
    for vertex in snapping_region:
        for next_vertex in mesh.adjacency_list[vertex]:
            if next_vertex not in snapping_region:
                boundary_vertices.add(vertex)

    boundary_loop = set()
    for vertex in boundary_vertices:
        for next_vertex in mesh.adjacency_list[vertex]:
            if next_vertex in boundary_vertices:
                boundary_loop.add(Edge(vertex, next_vertex))
                if np.linalg.norm(mesh.positions[vertex] - mesh.positions[next_vertex]) <= .00000000001:
                    print("ZERO edge in boundary")

    boundary_vertices_index_map = {}
    for edge in boundary_loop:
        for vertex in mesh.adjacency_list[edge[0]] & mesh.adjacency_list[edge[1]]:
            if vertex in snapping_region:
                continue

            # add vertices if necessary
            if edge[0] in boundary_vertices_index_map:
                v1 = boundary_vertices_index_map[edge[0]]
            else:
                v1 = len(new_mesh.positions)
                new_mesh.positions.append(mesh.positions[edge[0]])

            if edge[1] in boundary_vertices_index_map:
                v2 = boundary_vertices_index_map[edge[1]]
            else:
                v2 = len(new_mesh.positions)
                new_mesh.positions.append(mesh.positions[edge[1]])

            # connect vertices
            if v1 not in new_mesh.adjacency_list:
                new_mesh.adjacency_list[v1] = set()
            if v2 not in new_mesh.adjacency_list:
                new_mesh.adjacency_list[v2] = set()
            new_mesh.adjacency_list[v1].add(v2)
            new_mesh.adjacency_list[v2].add(v1)

            # add edge to boundaries queue
            add_edge_to_boundaries_(boundaries, Edge(v1, v2), mesh.positions[vertex], new_mesh, point_cloud)

def find_closest_boundary(vertex_position, boundaries, mesh):
    # vertex_position = mesh.positions[vertex]
    closest = (math.inf, 0)
    for edge in boundaries:
        v1 = mesh.positions[edge[0]]
        v2 = mesh.positions[edge[1]]
        length_squared = np.dot(v1-v2, v1-v2)
        if length_squared == 0:
            # raise ValueError("Zero-length edges are bad")
            print("ZERO length edge D:")
            projection = v1
        else:
            # calculate fraction of distance from v1 to v2 of the point that's closest to vertex
            projection_t = np.dot(vertex_position - v1, v2 - v1) / length_squared
            projection_t = np.clip(projection_t, 0, 1)

            # project vertex onto edge
            projection = v1 + projection_t * (v2 - v1)

        distance = np.linalg.norm(projection - vertex_position)
        if distance < closest[0]:
            closest_vertex = edge[0] if projection_t < .5 else edge[1]
            closest = (distance, closest_vertex)
    return closest

def remesh(mesh1, mesh2, snapping_region1, snapping_region2, smoothing_factor=1, osculating_circle_angle_subtended=pi/4):
    point_cloud = PointCloud([(mesh1, snapping_region1), (mesh2, snapping_region2)], smoothing_factor, osculating_circle_angle_subtended)

    boundaries = UpdateablePriorityQueue()
    new_mesh = Mesh()
    # new_mesh.positions.append(np.array([0,0,0]))
    # initialize new_mesh and boundaries with snapping regions' boundaries
    add_snapping_region_boundary(boundaries, new_mesh, mesh1, snapping_region1, point_cloud)
    add_snapping_region_boundary(boundaries, new_mesh, mesh2, snapping_region2, point_cloud)

    it = 0
    while boundaries:
        it += 1
        if it % 20 == 0:
            save_mesh(new_mesh, it/20)
        (is_deferred, priority, vertex), edge = boundaries.pop()
        if priority < .1:
            # priority < 0 only set for cuts
            cut_ear(new_mesh, edge, vertex, boundaries, point_cloud)
            continue
        # vertex = predict_vertex(edge, field, other_vertex)

        # closest_dist, closest_vertex = find_closest_boundary(vertex, boundaries, new_mesh)
        closest1 = find_closest_boundary(vertex, boundaries, new_mesh)
        closest2 = (math.inf,0)
        for i,pos in enumerate(new_mesh.positions):
            dist = np.linalg.norm(pos - vertex)
            if dist < closest2[0]:
                closest2 = (dist, i)
        closest_dist, closest_vertex = min(closest1, closest2)
        if closest_dist < point_cloud.guidance_field(vertex) / 2:
            if is_deferred:
                print("merge")
                # create triangle with closest vertex of closest_edge
                # edge1_length = np.linalg.norm(new_mesh.positions[closest_edge[0]] - vertex)
                # edge2_length = np.linalg.norm(new_mesh.positions[closest_edge[1]] - vertex)

                # vertex_index = closest_edge[0] if edge1_length < edge2_length else closest_edge[1]
                vertex_index = closest_vertex
                connect_triangle(new_mesh, edge, vertex_index, boundaries, point_cloud)
                # if vertex closer to edge endpoints than closest_edge endpoints:
                #     split closest_edge
                # else:
                #     merge to closest_edge endpoint
            else:
                boundaries.push(edge, (True, priority, vertex))
        else:
            grow_triangle(new_mesh, edge, vertex, boundaries, point_cloud)

    # TODO return maps from old to new vertices on snapping region boundary
    return new_mesh

def save_mesh(mesh, n):
    lists = [tuple(i) for i in mesh.positions]
    # positions_element = PlyElement.describe(, "vertex")
    positions_element = PlyElement.describe(np.array(lists, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]), "vertex")

    faces = [(face,) for face in mesh.faces]
    faces_element = PlyElement.describe(np.array(faces, dtype=[("vertex_indices", "i4", (3,))]), "face")

    PlyData([positions_element, faces_element], text=True).write("progress" + str(n) + ".ply")
