import os
import numpy as np
from plyfile import PlyData
from scannet_utils import export_mesh

def align_axis(use_alignment=False, save_aligned_mesh=False):

    for base, dirnames, filenames in os.walk('./scans'):
        for f in filenames:
            if f.endswith('_vh_clean_2.ply') and not f.startswith('.'):

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
                            break

                    if axis_align_matrix != None:
                        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
                        pts = np.ones((vertices.shape[0], 4))
                        pts[:, 0:3] = vertices[:, 0:3]
                        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
                        vertices[:, 0:3] = pts[:, 0:3]
                    else:
                        print("No axis alignment matrix found")
                print(os.path.join(base, f.replace('_vh_clean_2.ply', '_axis_aligned.ply')))
                if save_aligned_mesh:
                    mesh = export_mesh(vertices, faces)
                    mesh.write(os.path.join(base, f.replace('_vh_clean_2.ply', '_axis_aligned.ply')))

if __name__ == '__main__':
    align_axis(use_alignment=True, save_aligned_mesh=True)
