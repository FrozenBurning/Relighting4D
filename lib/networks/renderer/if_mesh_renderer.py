import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
import numpy as np
import mcubes
import trimesh
from . import if_clight_renderer


class Renderer(if_clight_renderer.Renderer):
    def __init__(self, net):
        super(Renderer, self).__init__(net)

    def batchify_rays(self, wpts, alpha_decoder, chunk=1024 * 32):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_batch, n_point = wpts.shape[:2]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = alpha_decoder(wpts[:, i:i + chunk])
            all_ret.append(ret)
        all_ret = torch.cat(all_ret, 1)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        inside = batch['inside'][0].bool()
        pts = pts[0][inside][None]

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)
        alpha_decoder = lambda x: self.net.calculate_density(
            x, feature_volume, sp_input)

        alpha = self.batchify_rays(pts, alpha_decoder, 2048 * 64)

        alpha = alpha[0, :, 0].detach().cpu().numpy()
        cube = np.zeros(sh[1:-1])
        inside = inside.detach().cpu().numpy()
        cube[inside == 1] = alpha

        cube = np.pad(cube, 10, mode='constant')
        vertices, triangles = mcubes.marching_cubes(cube, cfg.mesh_th)
        # normals = compute_normal(vertices, triangles)

        # vertices = (vertices - 10) * 0.005
        # vertices = vertices + batch['wbounds'][0, 0].detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices, triangles)

        ret = {'cube': cube, 'mesh': mesh}

        return ret

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm