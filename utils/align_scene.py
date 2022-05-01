import os
import numpy as np
from plyfile import PlyData, PlyElement
from lib.config import CONF
def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")),
            ("y", np.dtype("float32")),
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")

    return PlyData([vertices, faces])

def align_axis(use_alignment=False, save_aligned_mesh=False):

    for base, dirnames, filenames in os.walk(CONF.PATH.SCANNET_SCANS):
        for f in filenames:
            if f.endswith('_vh_clean_2.ply') and not f.startswith('.'):
                if f.split('_vh_clean_2.ply')[0] == 'scene0591_02':
                    print(f)
                else:
                    continue
                plydata = PlyData.read(os.path.join(base,f))
                num_verts = plydata['vertex'].count
                vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
                vertices[:, 0] = plydata['vertex'].data['x']
                vertices[:, 1] = plydata['vertex'].data['y']
                vertices[:, 2] = plydata['vertex'].data['z']
                vertices[:, 3] = plydata['vertex'].data['red']
                vertices[:, 4] = plydata['vertex'].data['green']
                vertices[:, 5] = plydata['vertex'].data['blue']
                faces = plydata['face']

                if use_alignment:
                    meta_file = os.path.join(base, f.replace('_vh_clean_2.ply', '.txt'))
                    lines = open(meta_file).readlines()
                    axis_align_matrix = None
                    for line in lines:
                        if 'axisAlignment' in line:
                            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]

                    if axis_align_matrix != None:
                        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                        pts = np.ones((vertices.shape[0], 4))
                        pts[:, 0:3] = vertices[:, 0:3]
                        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
                        vertices[:, 0:3] = pts[:, 0:3]
                    else:
                        print("No axis alignment matrix found")
                if save_aligned_mesh:
                    mesh = export_mesh(vertices, faces)
                    print(os.path.join(base, f.replace('_vh_clean_2.ply', '_axis_aligned.ply')))
                    mesh.write(os.path.join(base, f.replace('_vh_clean_2.ply', '_axis_aligned.ply')))

if __name__ == '__main__':
    align_axis(use_alignment=True, save_aligned_mesh=True)
