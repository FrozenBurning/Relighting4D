
from os.path import basename
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp

from brdf.microfacet import Microfacet
from lib.networks.shape import Network as ShapeModel
from lib.networks.mlp import MLP
from lib.config import cfg
from lib.utils.img_utils import one_hot_img, func_linear2srgb, safe_acos, read_hdr


class Network(ShapeModel):
    def __init__(self):
        # BRDF
        self.pred_brdf = cfg.pred_brdf
        self.z_dim = 1 # scalar roughness in microfacet
        self.normalize_brdf_z = False
        # Shape
        self.shape_mode = cfg.shape_mode

        # By now we have all attributes required by grandparent init.
        super(Network, self).__init__()

        self.normal_smooth_weight = cfg.normal_smooth_weight
        self.lvis_smooth_weight = cfg.lvis_smooth_weight
        # BRDF
        self.albedo_smooth_weight = cfg.albedo_smooth_weight
        self.brdf_smooth_weight = cfg.brdf_smooth_weight
        # Lighting
        light_h = cfg.light_h
        self.light_res = (light_h, 2*light_h)
        lxyz, lareas = self._gen_lights()
        self.lxyz, self.lareas = lxyz, lareas
        maxv = cfg.light_init_max
        if cfg.achro_light:
            light = torch.randn(self.light_res + (1,))*maxv
        else:
            light = torch.randn(self.light_res + (3,))*maxv
        self._light = nn.parameter.Parameter(light)
        # Novel lighting conditions for relighting at test time:
        # (1) OLAT
        novel_olat = OrderedDict()
        light_shape = self.light_res + (3,)
        olat_inten = cfg.olat_inten
        ambient_inten = cfg.ambient_inten
        ambient = ambient_inten*torch.ones(light_shape, device=torch.device('cuda:0'))
        olats = [27, 91, 149, 200, 288, 333, 398, 488]
        idx = -1
        for i in range(self.light_res[0]):
            for j in range(self.light_res[1]):
                idx += 1
                if idx in olats:
                    one_hot = one_hot_img(*ambient.shape, i, j)
                    envmap = olat_inten * one_hot + ambient
                    novel_olat['%04d-%04d' % (i, j)] = envmap
        self.novel_olat = novel_olat
        # (2) Light probes
        novel_probes = OrderedDict()
        for path in sorted(os.listdir('light-probes/')):
            if '.hdr' in path:
                name = basename(path)[:-len('.hdr')]
                arr = read_hdr(osp.join('light-probes/', path))
                tensor = torch.from_numpy(arr).cuda()
                novel_probes[name] = tensor
        self.novel_probes = novel_probes

    def _init_embedder(self):
        # Use grandparent's embedders, not parent's, since we don't need
        # the embedder for BRDF coordinates
        embedder = super()._init_embedder()
        return embedder

    def _init_net(self):
        net = super()._init_net()
        mlp_width = cfg.mlp_width
        mlp_depth = cfg.mlp_depth
        mlp_skip_at = cfg.mlp_skip_at
        smpl_feature_dim = 256
        # Override the roughness MLP output layer to add sigmoid so that [0, 1]
        if cfg.use_xyz:
            net['albedo_mlp'] = MLP(
                smpl_feature_dim + 2*cfg.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
            )
        else:
            net['albedo_mlp'] = MLP(
                smpl_feature_dim, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
            )
        net['albedo_out'] = MLP(
            mlp_width, [3], act=['sigmoid']
        )
        # brdf
        if self.pred_brdf:
            if cfg.use_xyz:
                net['brdf_z_mlp'] = MLP(
                    smpl_feature_dim + 2*cfg.n_freqs_xyz*3+3, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
                )
            else:
                net['brdf_z_mlp'] = MLP(
                    smpl_feature_dim, [mlp_width]*mlp_depth, act=['relu']*mlp_depth, skip_at=[mlp_skip_at]
                )
            net['brdf_z_out'] = MLP(mlp_width, [self.z_dim], act=['sigmoid']) # [0, 1]
        return net

    def _eval_brdf_at(self, pts2l, pts2c, normal, albedo, brdf_prop):
        """Fixed to microfacet (GGX).
        """
        rough = brdf_prop
        fresnel_f0 = cfg.fresnel_f0
        microfacet = Microfacet(f0=fresnel_f0)
        brdf = microfacet(pts2l, pts2c, normal, albedo=albedo, rough=rough)
        return brdf # NxLx3

    def _brdf_prop_as_img(self, brdf_prop):
        """Roughness in the microfacet BRDF.

        Input and output are both NumPy arrays, not tensors.
        """
        z_rgb = np.concatenate([brdf_prop] * 3, axis=2)
        return z_rgb

    def light(self):
        # No negative light
        if cfg.fix_light:
            return torch.ones(self.light_res + (3,)).to(self._light) + self._light.repeat(1,1,3)
        else:
            if cfg.perturb_light > 0:
                light_noise = torch.normal(mean=0, std=self._light.max().item() / cfg.perturb_light, size=self.light_res + (3,)).to(self._light)
            else:
                light_noise = 0.
            if cfg.achro_light:
                return torch.clip(self._light.repeat(1, 1, 3), min=0., max=1e6) + light_noise
            else:
                return torch.clip(self._light, min=0., max=1e6) + light_noise


    def forward(self, batch, mode='train', relight_olat=False, relight_probes=False, albedo_scale=None, albedo_override=None, brdf_z_override=None):
        xyz_jitter_std = cfg.xyz_jitter_std
        id_, hw, rayo, _, rgb, alpha, xyz, normal, lvis, raw_batch = batch
        features = None
        features_jitter = None
        if mode == 'val':
            self.eval()
            xyz_jitter_std = 0.

        # jitter
        if xyz_jitter_std > 0:
            xyz_noise = torch.normal(mean=0, std=xyz_jitter_std, size=xyz.shape).to(xyz)
        else:
            xyz_noise = None

        wpts = xyz
        sp_input = raw_batch['sp_input']
        feature_volume = raw_batch['feature_volume']
        with torch.no_grad():
            features = self.net['neuralbody'].get_feature(wpts, sp_input, feature_volume)
            features = features.transpose(1, 2)
            if xyz_noise is not None:
                features_jitter = self.net['neuralbody'].get_feature(wpts + xyz_noise, sp_input, feature_volume)
                features_jitter = features_jitter.transpose(1, 2)

        features = self.net['latent_fc'](features).reshape(-1, 256)
        if xyz_noise is not None:
            features_jitter = self.net['latent_fc'](features_jitter).reshape(-1, 256)

        surf2light = self._get_ldir(xyz)
        surf2cam = self._get_vdir(rayo.float(), xyz)

        # normals
        if self.shape_mode == 'nerf':
            normal_pred = normal.reshape(-1, 3)
            normal_jitter = None
        else:
            normal_pred = self._pred_normal_at(xyz, features)
            if xyz_noise is None:
                normal_jitter = None
            else:
                normal_jitter = self._pred_normal_at(xyz + xyz_noise, features_jitter)
        normal_pred = torch.nn.functional.normalize(normal_pred, p=2, dim=-1, eps=1e-7)
        if normal_jitter is not None:
            normal_jitter = torch.nn.functional.normalize(normal_jitter, p=2, dim=-1, eps=1e-7)

        # light visibility
        if self.shape_mode == 'nerf':
            lvis_pred = torch.clip(lvis.reshape(-1, 512), 1e-7, 1.)
            lvis_jitter = None
        else:
            lvis_pred = self._pred_lvis_at(xyz, surf2light, features)
            if xyz_noise is None:
                lvis_jitter = None
            else:
                lvis_jitter = self._pred_lvis_at(xyz + xyz_noise, surf2light, features_jitter)
        
        # albedo
        albedo = self._pred_albedo_at(xyz, features)
        if xyz_noise is None:
            albedo_jitter = None
        else:
            albedo_jitter = self._pred_albedo_at(xyz + xyz_noise, features_jitter)
        
        if albedo_scale is not None:
            raise NotImplementedError

        if albedo_override is not None:
            raise NotImplementedError

        if self.pred_brdf:
            brdf_prop = self._pred_brdf_at(xyz, features)
            if xyz_noise is None:
                brdf_prop_jitter = None
            else:
                brdf_prop_jitter = self._pred_brdf_at(xyz + xyz_noise, features_jitter)
            if self.normalize_brdf_z:
                brdf_prop = torch.nn.functional.normalize(brdf_prop, p=2, dim=-1, eps=1e-7)
                if brdf_prop_jitter is not None:
                    brdf_prop_jitter = torch.nn.functional.normalize(brdf_prop_jitter, p=2, dim=-1, eps=1e-7)
        else:
            brdf_prop = self._get_default_brdf_at(xyz)
            brdf_prop_jitter = None
        
        if brdf_z_override is not None:
            raise NotImplementedError
        brdf = self._eval_brdf_at(surf2light, surf2cam, normal_pred, albedo, brdf_prop) # NxLx3

        # rendering
        rgb_pred, rgb_olat, rgb_probes, hdr = self._render(lvis_pred, brdf, surf2light, normal_pred, relight_olat=relight_olat, relight_probes=relight_probes)

        pred = {
            'rgb': rgb_pred, 'normal': normal_pred, 'lvis': lvis_pred,
            'albedo': albedo, 'brdf': brdf_prop, 'hdr': hdr}
        if rgb_olat is not None:
            pred['rgb_olat'] = rgb_olat
        if rgb_probes is not None:
            pred['rgb_probes'] = rgb_probes
        gt = {'rgb': rgb, 'normal': normal, 'lvis': lvis, 'alpha': alpha}
        loss_kwargs = {
            'mode': mode, 'normal_jitter': normal_jitter,
            'lvis_jitter': lvis_jitter, 'brdf_prop_jitter': brdf_prop_jitter,
            'albedo_jitter': albedo_jitter}
        # ------ To visualize
        to_vis = {'id': id_, 'hw': hw}
        for k, v in pred.items():
            to_vis['pred_' + k] = v
        for k, v in gt.items():
            to_vis['gt_' + k] = v
        return pred, gt, loss_kwargs, to_vis

    def _pred_albedo_at(self, pts, features=None):
        albedo_scale = cfg.albedo_slope
        albedo_bias = cfg.albedo_bias
        mlp = self.net['albedo_mlp']
        out = self.net['albedo_out']
        embedder = self.embedder['xyz']
        scaled_pts = self.xyz_scale * pts
        surf_embed = embedder.embed(scaled_pts.reshape(-1, 3)).float()
        if features is not None:
            if cfg.use_xyz:
                surf_embed = torch.cat([features, surf_embed], dim=-1)
            else:
                surf_embed = features

        def chunk_func(surf):
            albedo = out(mlp(surf))
            return albedo
        
        albedo = self.chunk_apply(chunk_func, surf_embed, 3, self.mlp_chunk)
        albedo = albedo_scale * albedo + albedo_bias # [bias, scale + bias]
        return albedo # Nx3

    def _pred_brdf_at(self, pts, features=None):
        mlp = self.net['brdf_z_mlp']
        out = self.net['brdf_z_out']
        embedder = self.embedder['xyz']
        scaled_pts = self.xyz_scale * pts
        surf_embed = embedder.embed(scaled_pts.reshape(-1, 3)).float()
        if features is not None:
            if cfg.use_xyz:
                surf_embed = torch.cat([features, surf_embed], dim=-1)
            else:
                surf_embed = features

        def chunk_func(surf):
            brdf_z = out(mlp(surf))
            return brdf_z

        brdf_z = self.chunk_apply(chunk_func, surf_embed, self.z_dim, self.mlp_chunk)
        return brdf_z # NxZ

    def _render(
            self, light_vis, brdf, surf2light, normal,
            relight_olat=False, relight_probes=False,
            white_light_override=False, white_lvis_override=False):
        linear2srgb = cfg.linear2srgb
        light = self.light()

        lcos = torch.einsum('ijk,ik->ij', surf2light, normal)
        areas = self.lareas.reshape(1, -1, 1)
        front_lit = lcos > 0
        lvis = front_lit * light_vis

        hdr = None
        # hdr_contrib = brdf * lcos[:, :, None] * areas
        # hdr = torch.sum(hdr_contrib, dim=1)

        def integrate(light):
            light_flat = light.reshape(-1, 3)
            light = lvis[:, :, None] * light_flat[None, :, :] # NxLx3
            light_pix_contrib = brdf * light * lcos[:, :, None] * areas # NxLx3
            rgb = torch.sum(light_pix_contrib, dim=1) #Nx3
            # Tonemapping
            rgb = torch.clip(rgb, 0., 1.)
            # Colorspace transform
            if linear2srgb:
                rgb = func_linear2srgb(rgb)
            return rgb
        
        rgb = integrate(light)
        # print('light', light)
        rgb_olat = None
        if relight_olat:
            rgb_olat = []
            for _, light in self.novel_olat.items():
                rgb_relit = integrate(light)
                rgb_olat.append(rgb_relit)
            rgb_olat = torch.cat([x[:, None, :] for x in rgb_olat], dim=1)

        rgb_probes = None
        if relight_probes:
            rgb_probes = []
            for _, light in self.novel_probes.items():
                rgb_relit = integrate(0.25*light + 0.1*self.light())
                rgb_probes.append(rgb_relit)
            rgb_probes = torch.cat([x[:, None, :] for x in rgb_probes], dim=1)
        return rgb, rgb_olat, rgb_probes, hdr # Nx3
    
    def compute_loss(self, pred, gt, **kwargs):
        normal_loss_weight = cfg.normal_loss_weight
        lvis_loss_weight = cfg.lvis_loss_weight
        smooth_use_l1 = cfg.smooth_use_l1
        light_tv_weight = cfg.light_tv_weight
        light_achro_weight = cfg.light_achro_weight
        smooth_loss = nn.L1Loss() if smooth_use_l1 else nn.MSELoss()
        mode = kwargs.pop('mode')
        normal_jitter = kwargs.pop('normal_jitter')
        lvis_jitter = kwargs.pop('lvis_jitter')
        albedo_jitter = kwargs.pop('albedo_jitter')
        brdf_prop_jitter = kwargs.pop('brdf_prop_jitter')

        alpha, rgb_gt = gt['alpha'], gt['rgb']
        rgb_pred = pred['rgb']
        normal_pred, normal_gt = pred['normal'], gt['normal']
        lvis_pred, lvis_gt = pred['lvis'], gt['lvis']
        albedo_pred = pred['albedo']
        brdf_prop_pred = pred['brdf']
        hdr = pred['hdr']

        # RGB recon. loss is always here
        loss = 0
        mse = nn.MSELoss()
        rgb_loss = mse(rgb_gt.reshape(-1, 3), rgb_pred) * 10.
        loss = loss + rgb_loss # N
        # If validation, just MSE -- return immediately
        if mode == 'vali':
            return loss
        # If we modify the geometry
        normal_loss = torch.zeros(1, device=torch.device('cuda:0'))
        lvis_loss = torch.zeros(1, device=torch.device('cuda:0'))
        normal_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        lvis_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        albedo_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        brdf_smooth_loss = torch.zeros(1, device=torch.device('cuda:0'))
        albedo_entropy = torch.zeros(1, device=torch.device('cuda:0'))
        if self.shape_mode in ('scratch', 'finetune'):
            # Predicted values should be close to initial values
            normal_loss = mse(normal_gt.reshape(-1, 3), normal_pred) # N
            lvis_loss = mse(lvis_gt.reshape(-1, 512), lvis_pred) # N
            if self.shape_mode in ('scratch'):
                assert cfg.use_shape_sup
            if cfg.use_shape_sup:
                loss += normal_loss_weight * normal_loss
                loss += lvis_loss_weight * lvis_loss
            # Predicted values should be smooth
            if normal_jitter is not None:
                normal_smooth_loss = smooth_loss(normal_pred, normal_jitter) # N
                loss += self.normal_smooth_weight * normal_smooth_loss
            if lvis_jitter is not None:
                lvis_smooth_loss = smooth_loss(lvis_pred, lvis_jitter) # N
                loss += self.lvis_smooth_weight * lvis_smooth_loss
        # Albedo should be smooth
        if albedo_jitter is not None:
            albedo_smooth_loss = smooth_loss(albedo_pred, albedo_jitter) # N
            loss += self.albedo_smooth_weight * albedo_smooth_loss
        # BRDF property should be smooth
        if brdf_prop_jitter is not None:
            brdf_smooth_loss = smooth_loss(brdf_prop_pred, brdf_prop_jitter) # N
            loss += self.brdf_smooth_weight * brdf_smooth_loss

        # Light should be smooth
        if mode == 'train':
            light = self.light()
            # Spatial TV penalty
            if light_tv_weight > 0:
                dx = light - torch.roll(light, 1, 1)
                dy = light - torch.roll(light, 1, 0)
                tv = torch.sum(dx ** 2 + dy ** 2)
                loss += light_tv_weight * tv
            # Cross-channel TV penalty
            if light_achro_weight > 0:
                dc = light - torch.roll(light, 1, 2)
                tv = torch.sum(dc ** 2)
                loss += light_achro_weight * tv

        if cfg.albedo_sparsity > 0:
            albedo_entropy = 0
            for i in range(3):
                channel = albedo_pred[..., i]
                hist = GaussianHistogram(15, 0., 1., sigma=torch.var(channel))
                h = hist(channel)
                if h.sum() > 1e-6:
                    h = h.div(h.sum()) + 1e-6
                else:
                    h = torch.ones_like(h).to(h)
                albedo_entropy += torch.sum(-h*torch.log(h))
            loss += cfg.albedo_sparsity * albedo_entropy

        return loss, normal_loss_weight*normal_loss, lvis_loss_weight*lvis_loss, self.normal_smooth_weight*normal_smooth_loss, self.lvis_smooth_weight*lvis_smooth_loss, self.albedo_smooth_weight*albedo_smooth_loss, self.brdf_smooth_weight*brdf_smooth_loss, rgb_loss, albedo_entropy

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=torch.device('cuda:0')).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x