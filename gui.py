from math import pi, cos, sin
from snappaste import merge, save_mesh
from plyfile import PlyData
from mesh import Mesh
from trimesh import Trimesh
import numpy as np

def validate_args(args, value_name):
    if len(args) != 3:
        raise ValueError("bad arg count")

    mesh, axis, value = args

    if len(axis) > 1:
        raise ValueError("invalid axis")
    axis = ord(axis) - ord("x")
    if axis < 0 or axis > 2:
        raise ValueError("invalid axis")

    if mesh != "1" and mesh != "2":
        raise ValueError("invalid mesh")
    mesh = int(mesh)

    try:
        value = float(value)
    except ValueError:
        raise ValueError("invalid " + value_name)

    return mesh, axis, value

def validate_args_scale(args, value_name):
    if len(args) < 2:
        raise ValueError("bad arg count")

    if len(args) == 2:
        mesh, value = args
        axis = None
    else:
        mesh, axis, value = args
        if len(axis) > 1:
            raise ValueError("invalid axis")
        axis = ord(axis) - ord("x")
        if axis < 0 or axis > 2:
            raise ValueError("invalid axis")

    if mesh != "1" and mesh != "2":
        raise ValueError("invalid mesh")
    mesh = int(mesh)

    try:
        value = float(value)
    except ValueError:
        raise ValueError("invalid " + value_name)

    return mesh, axis, value

def main():
    mesh1 = input("mesh 1 filename: ") + ".ply"
    mesh1 = Mesh(PlyData.read(mesh1))

    mesh2 = input("mesh 2 filename: ") + ".ply"
    mesh2 = Mesh(PlyData.read(mesh2))

    result = None

    total_translation = np.zeros(3)

    while(True):
        command, *args = input("command: ").split()
        try:
            if command == "preview" or command == "p":
                if len(args) == 0:
                    t1 = Trimesh(vertices=mesh1.positions, faces=mesh1.faces)
                    t2 = Trimesh(vertices=mesh2.positions, faces=mesh2.faces)
                    (t1+t2).show()
                elif args[0] == "1":
                    Trimesh(vertices=mesh1.positions, faces=mesh1.faces).show()
                elif args[0] == "2":
                    Trimesh(vertices=mesh2.positions, faces=mesh2.faces).show()
                else:
                    raise ValueError("invalid mesh")
            elif command == "translate" or command == "t":
                mesh, axis, distance = validate_args(args, "distance")
                
                mesh = mesh1 if mesh == 1 else mesh2

                translation = np.zeros(3)
                translation[axis] = distance
                total_translation += translation
                for position in mesh.positions:
                    position += translation
            elif command == "rotate" or command == "r":
                mesh, axis, angle = validate_args(args, "angle")
                angle *= pi / 180
                a = cos(angle)
                c = sin(angle)
                b = -c
                d = a
                rotation = np.eye(3,3)
                i = axis - 2
                rotation[i, i] = a
                rotation[i, i+1] = b
                rotation[i+1, i] = c
                rotation[i+1, i+1] = d

                mesh = mesh1 if mesh == 1 else mesh2

                for i,position in enumerate(mesh.positions):
                    mesh.positions[i] = np.dot(rotation, (position - total_translation)) + total_translation
            elif command == "scale" or command == "s":
                mesh, axis, factor = validate_args_scale(args, "factor")

                mesh = mesh1 if mesh == 1 else mesh2

                scaling = np.ones(3)
                if axis is None:
                    scaling *= factor
                else:
                    scaling[axis] = factor

                for position in mesh.positions:
                    position *= scaling
            elif command == "merge":
                if len(args) > 1:
                    raise ValueError("bad arg count")
                if len(args) == 1:
                    try:
                        iterations = int(args[0])
                    except ValueError:
                        raise ValueError("bad iteration count")
                else:
                    iterations = 6
                elasticity = 1
                smoothing_factor = 1
                result = merge(mesh1, mesh2, iterations, elasticity, smoothing_factor)
                Trimesh(vertices=result.positions, faces=result.faces).show()
            elif command == "show":
                if result is None:
                    raise ValueError("not merged yet")
                Trimesh(vertices=result.positions, faces=result.faces).show()
            elif command == "save": # [name], or [mesh] [name]
                if len(args) < 1:
                    raise ValueError("bad arg count")
                if len(args) > 1:
                    if args[0] == "1":
                        save_mesh(mesh1, args[1] + ".ply")
                    elif args[0] == "2":
                        save_mesh(mesh2, args[1] + ".ply")
                    elif args[0] == "both":
                        save_mesh(mesh1, args[1] + "1.ply")
                        save_mesh(mesh2, args[1] + "2.ply")
                    else:
                        raise ValueError("invalid mesh")
                else:
                    if result is None:
                        raise ValueError("not merged yet")
                    save_mesh(result, args[0] + ".ply", np_array=False)
            elif command == "exit":
                break
            else:
                raise ValueError("bad command")
        except ValueError as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
