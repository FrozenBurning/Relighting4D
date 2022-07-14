import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from brdf.renderer import gen_light_xyz
from lib.networks.embedder import Embedder
from lib.networks.latent_xyzc import Network as NeuralBody
from lib.networks.mlp import MLP
from lib.config import cfg


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.mlp_chunk = cfg.mlp_chunk
        self.embedder = self._init_embedder()
        self.net = self._init_net()

        self.xyz_scale = cfg.xyz_scale
        light_xyz, _ = self._gen_lights()
        self.light_xyz = light_xyz.reshape(1,-1,3)

    def _gen_lights(self):
        light_h = int(cfg.light_h)
        light_w = int(light_h * 2)
        lxyz, lareas = gen_light_xyz(light_h, light_w)
        if 'xyzc' in cfg.exp_name or 'zju' in cfg.exp_name:
            print("zju-mocap:{}, using different lighting......".format(cfg.exp_name))
            lxyz[...,1] = -lxyz[...,1]
            lxyz[...,0] = -lxyz[...,0]
        else:
            lxyz = lxyz[..., [0, 2, 1]]
            lxyz[...,1] = -lxyz[...,1]
        lxyz = torch.from_numpy(lxyz).float().cuda()
        lareas = torch.from_numpy(lareas).float().cuda()
        return lxyz, lareas
    
    def _init_net(self):
        mlp_width = cfg.mlp_width
        mlp_depth = cfg.mlp_depth
        mlp_skip_at = cfg.mlp_skip_at
        net = nn.ModuleDict()
        smpl_feature_dim = 0
        assert cfg.smpl_model_ckpt
        net['neuralbody'] = NeuralBody()
        net['neuralbody'].load_state_dict(torch.load(cfg.smpl_model_ckpt)['net'])
        net['neuralbody'].eval()
        for param in net['neuralbody'].parameters():
            param.requires_grad = False
        net['latent_fc'] = nn.Linear(384, 256)
        smpl_feature_dim = 256
        # Normals
        net['normal_mlp'] = MLP(
            smpl_feature_dim + 2*cfg.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
        )
        net['normal_out'] = MLP(
            mlp_width, [3], act=None
        )
        net['lvis_mlp'] = MLP(
            smpl_feature_dim + 2*cfg.n_freqs_xyz*3+3+2*cfg.n_freqs_ldir*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
        )
        net['lvis_out'] = MLP(
            mlp_width, [1], act=['sigmoid']
        )
        return net

    def _init_embedder(self):
        kwargs = {
            'input_dims': 3,
            'include_input': True,
            'max_freq_log2': cfg.n_freqs_xyz - 1,
            'num_freqs': cfg.n_freqs_xyz,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos]
        }
        embedder_xyz = Embedder(**kwargs)
        kwargs['max_freq_log2'] = cfg.n_freqs_ldir - 1
        kwargs['num_freqs'] = cfg.n_freqs_ldir
        embedder_ldir = Embedder(**kwargs)
        kwargs['max_freq_log2'] = cfg.n_freqs_vdir - 1
        kwargs['num_freqs'] = cfg.n_freqs_vdir
        embedder_vdir = Embedder(**kwargs)
        embedder = {
            'xyz': embedder_xyz, 'ldir': embedder_ldir, 'vdir': embedder_vdir
        }
        return embedder

    def _get_ldir(self, pts):
        bs, n_lights, _ = self.light_xyz.shape
        _, n_rays, _ = pts.shape
        surf2light = self.light_xyz.reshape(1, -1, 3) - pts.reshape(n_rays, 3)[:, None, :]
        surf2light = torch.nn.functional.normalize(surf2light, p=2, dim=-1, eps=1e-7)
        return surf2light

    @staticmethod
    def _get_vdir(cam_loc, pts):
        surf2cam = cam_loc - pts
        surf2cam = torch.nn.functional.normalize(surf2cam, p=2, dim=-1, eps=1e-7)
        return surf2cam.reshape(-1, 3) # Nx3


    def forward(self, batch, mode='train'):
        id_, hw, alpha, xyz, normal, lvis, light_map = batch

        surf2light = self._get_ldir(xyz)

        normal_pred = self._pred_normal_at(xyz)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1, eps=1e-7)

        lvis_pred = self._pred_lvis_at(xyz, surf2light)
        # ------ Loss
        pred = {'normal': normal_pred, 'lvis': lvis_pred}
        gt = {'normal': normal.reshape(-1, 3), 'lvis': lvis.reshape(-1, 512), 'alpha': alpha}
        loss_kwargs = {}
        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v

        return pred, gt, loss_kwargs, to_vis

    @staticmethod
    def chunk_apply(func, x, dim, n_chunk):
        n = x.shape[0]
        ret = torch.zeros((n, dim)).to(x)
        for i in range(0, n, n_chunk):
            end_i = min(n, i + n_chunk)
            x_chunk = x[i:end_i]
            ret_chunk = func(x_chunk)
            ret[i:end_i] = ret_chunk
        return ret

    def _pred_normal_at(self, pts, features=None):
        eps = 1e-6
        mlp = self.net['normal_mlp']
        out = self.net['normal_out']
        scaled_pts = self.xyz_scale * pts
        surf_embed = self.embedder['xyz'].embed(scaled_pts.reshape(-1, 3)).float()
        if features is not None:
            surf_embed = torch.cat([features, surf_embed], dim=-1)

        def chunk_func(surf):
            normals = out(mlp(surf))
            return normals
        
        normal = self.chunk_apply(chunk_func, surf_embed, 3, self.mlp_chunk)
        normal = normal + eps
        return normal

    def _pred_lvis_at(self, pts, surf2light, features=None):
        mlp = self.net['lvis_mlp']
        out = self.net['lvis_out']
        scaled_pts = self.xyz_scale * pts
        n_lights = surf2light.shape[1]
        surf2light_flat = surf2light.reshape(-1, 3) #NLx3
        surf_rep = scaled_pts.reshape(-1, 3)[:, None, :].repeat(1, n_lights, 1)
        surf_flat = surf_rep.reshape(-1, 3)

        surf_embed = self.embedder['xyz'].embed(surf_flat).float()
        surf2light_embed = self.embedder['ldir'].embed(surf2light_flat).float()
        if features is not None:
            feat = features[:, None, :].repeat(1, n_lights, 1).reshape(-1, 256)
            mlp_input = torch.cat([feat, surf_embed, surf2light_embed], dim=-1)
        else:
            mlp_input = torch.cat([surf_embed, surf2light_embed], dim=-1)

        def chunk_func(input):
            lvis = out(mlp(input))
            return lvis

        lvis_flat = self.chunk_apply(chunk_func, mlp_input, 1, self.mlp_chunk)
        lvis = lvis_flat.reshape(scaled_pts.shape[1], n_lights)
        
        return lvis

    def compute_loss(self, pred, gt, **kwargs):
        normal_loss_weight = cfg.normal_loss_weight
        lvis_loss_weight = cfg.lvis_loss_weight

        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']

        alpha_map = gt['alpha']
        mse = nn.MSELoss()
        normal_loss = mse(normal_gt, normal_pred)
        lvis_loss = mse(lvis_gt, lvis_pred)
        loss = normal_loss * normal_loss_weight + lvis_loss * lvis_loss_weight
        return loss, normal_loss, lvis_loss

