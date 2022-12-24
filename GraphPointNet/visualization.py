from plyfile import PlyData, PlyElement
import numpy as np

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (255, 128, 0),
    (255, 0, 128),
    (0, 255, 128),
    (128, 255, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 255, 128),
    (255, 128, 255),
    (128, 255, 255),
    (192, 0, 0),
    (0, 192, 0),
    (0, 0, 192),
    (192, 192, 0),
    (192, 0, 192),
    (0, 192, 192),
    (255, 192, 0),
    (255, 0, 192),
    (0, 255, 192),
    (192, 255, 0),
    (192, 0, 255),
    (0, 192, 255),
    (255, 255, 192),
    (255, 192, 255),
    (192, 255, 255),
    (128, 64, 0),
    (128, 0, 64),
    (0, 128, 64),
    (64, 128, 0),
    (64, 0, 128),
    (0, 64, 128),
    (255, 128, 64),
    (255, 64, 128),
    (64, 255, 128),
    (128, 255, 64),
    (128, 64, 255),
    (64, 128, 255),
    (255, 255, 128),
    (255, 128, 255)
]


def visualize_point_cloud(points, part_seg, output_file):
    # Define the list of points and their colors
    # points_color = [(x1, y1, z1, r1, g1, b1), (x2, y2, z2, r2, g2, b2), ...]
    '''points: (N, 3)'''
    # swap dimension to (N, 3)
    points = np.transpose(points, (1,0))
    points_color = [(point[0], point[1], point[2], colors[seg][0], colors[seg][1], colors[seg][2]) for point, seg in zip(points, part_seg)]
    
    # Create a list of vertices (each vertex is a point with its color)
    vertices = []
    counter = 0
    # for point in points_color:
    #     x, y, z, r, g, b = point
    #     vertex = PlyElement.describe(
    #         np.array([(x, y, z, r, g, b)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
    #         f'vertex'
    #     )
    #     vertices.append(vertex)

    # # Create the PLY file with the vertices
    # PlyData(vertices).write(output_file)
    vertices = np.array([(x, y, z, r, g, b) for x, y, z, r, g, b in points_color], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices = PlyElement.describe(vertices, 'vertex')
    PlyData([vertices]).write(output_file)