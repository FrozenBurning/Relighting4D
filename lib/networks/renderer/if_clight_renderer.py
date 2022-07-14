import torch
import numpy as np
from lib.config import cfg
from .nerf_net_utils import *
from brdf.renderer import gen_light_xyz
from lib.networks.latent_xyzc import Network as NeuralBody


class Renderer:
    def __init__(self, net):
        self.net = net

    def get_sampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]
        return pts, z_vals

    def prepare_sp_input(self, batch):
        # feature, coordinate, shape, batch size
        sp_input = {}

        # coordinate: [N, 4], batch_idx, z, y, x
        sh = batch['coord'].shape
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]
        idx = torch.cat(idx).to(batch['coord'])
        coord = batch['coord'].view(-1, sh[-1])
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1)

        out_sh, _ = torch.max(batch['out_sh'], dim=0)
        sp_input['out_sh'] = out_sh.tolist()
        sp_input['batch_size'] = sh[0]

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']
        sp_input['R'] = batch['R']
        sp_input['Th'] = batch['Th']

        # used for color function
        sp_input['latent_index'] = batch['latent_index']

        return sp_input

    def get_density(self, wpts, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        raw = raw_decoder(wpts)
        return raw

    def get_density_color(self, wpts, viewdir, raw_decoder):
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume,
                        sp_input, batch):
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color(
            x_point, viewdir_val, feature_volume, sp_input)

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, raw_decoder)
        # [raw rgb and raw sigma]

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw = wpts_raw.reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret

    def render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               feature_volume, sp_input, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret

    def relighte2e_render(self, batch):
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # encode neural body
        sp_input = self.prepare_sp_input(batch)
        feature_volume = self.net.encode_sparse_voxels(sp_input)

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pts, z = self.get_sampling_points(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk)
            raw_decoder = lambda x_point: self.net.calculate_density(x_point, feature_volume, sp_input)
            # get density and normal
            pts.requires_grad = True
            density = self.get_density(pts, raw_decoder)
            normal = torch.autograd.grad(density, pts, torch.ones_like(density), retain_graph=True)[0]

            n_batch, n_pixel, n_sample = pts.shape[:3]
            raw = density.reshape(-1, n_sample, 1)
            z = z.reshape(-1, n_sample)
            _, disp, acc, weights, depth = density2outputs(raw, z, ray_d_chunk.reshape(-1, 3), cfg.raw_noise_std, cfg.white_bkgd)
            surf_chunk = ray_o_chunk + ray_d_chunk * depth[:, None]
            surf_chunk = surf_chunk.reshape(-1, 3)
            normal = torch.sum(weights[..., None] * normal, -2)
            normal = -torch.nn.functional.normalize(normal, p=2, dim=1)
            normal = normal.reshape(-1, 3)
            # get lvis
            with torch.no_grad():
                light_xyz, _ = gen_light_xyz(16, 32)
                if 'xyzc' in cfg.exp_name or 'zju' in cfg.exp_name:
                    print("zju-mocap:{}, using different lighting......".format(cfg.exp_name))
                    light_xyz[...,1] = -light_xyz[...,1]
                    light_xyz[...,0] = -light_xyz[...,0]
                else:
                    light_xyz = light_xyz[..., [0, 2, 1]]
                    light_xyz[...,1] = -light_xyz[...,1]
                light_xyz = torch.from_numpy(light_xyz).float().cuda().reshape(1,-1,3)
                n_lights = light_xyz.shape[1]
                lvis_hit = torch.zeros(surf_chunk.shape[0], n_lights).cuda()
                lpix_chunk = 64
                if cfg.lvis_far > 0.:
                    for i in range(0, n_lights, lpix_chunk):
                        end_i = min(n_lights, i + lpix_chunk)
                        lxyz_chunk = light_xyz[:, i:end_i, :]
                        surf2light = lxyz_chunk - surf_chunk[:, None, :]
                        surf2light = surf2light / torch.norm(surf2light, p=2, dim=2, keepdim=True)
                        surf2lightflat = surf2light.reshape(-1, 3)
                        surfrep = surf_chunk.repeat(1, surf2light.shape[1], 1)
                        surfflat = surfrep.reshape(-1, 3)
                        lcos = torch.einsum('ijk,ik->ij', surf2light, normal)
                        front_lit = lcos > 0.
                        front_lit = front_lit.reshape(-1)
                        if torch.sum(front_lit) == 0:
                            continue
                        front_surf = surfflat[front_lit, :].reshape(1, -1, 3)
                        front_surf2light = surf2lightflat[front_lit, :].reshape(1, -1, 3)
                        lvis_far = (torch.ones((1, front_surf.shape[0]))*cfg.lvis_far).cuda()
                        lvis_near = (torch.ones((1, front_surf.shape[0]))*cfg.lvis_near).cuda()
                        lvis_pts, lvis_z = self.get_sampling_points(front_surf, front_surf2light, lvis_near, lvis_far)
                        lvis_density = self.get_density(lvis_pts, raw_decoder)
                        _, _, lvis_sample = lvis_pts.shape[:3]
                        raw = lvis_density.reshape(-1, lvis_sample, 1)
                        lvis_z = lvis_z.reshape(-1, n_sample)
                        _, _, lvis_acc, _, _ = density2outputs(raw, lvis_z, front_surf2light.reshape(-1, 3), cfg.raw_noise_std, cfg.white_bkgd)
                        tmp = torch.zeros(lvis_hit.shape, dtype=bool)
                        front_lit = front_lit.reshape(n_pixel, lpix_chunk)
                        tmp[:, i:end_i] = front_lit
                        lvis_hit[tmp] = 1 - lvis_acc

            ret = {
                'surf': surf_chunk.reshape(n_batch, n_pixel, -1),
                'normal': normal.reshape(n_batch, n_pixel, -1),
                'lvis_hit': lvis_hit.reshape(n_batch, n_pixel, -1),
                'alpha_map': acc.reshape(n_batch, n_pixel),
            }
            ret_list.append(ret)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}
        ret.update({
            'sp_input': sp_input,
            'feature_volume': feature_volume,
        })

        return ret

    def relight_render(self, batch):
        idx = batch['idx']
        img_gt = batch['img_gt']
        img = batch['img']
        mask_at_box = batch['mask_at_box']
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        sh = ray_o.shape

        # encode neural body
        with torch.no_grad():
            sp_input = self.prepare_sp_input(batch)
            feature_volume = self.net.net['neuralbody'].encode_sparse_voxels(sp_input)
            batch.update({
                'sp_input': sp_input,
                'feature_volume': feature_volume,
            })


        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 4096
        rgb_list = []
        brdf_list = []
        normal_list = []
        relit_list = []
        hdr_relit_list = []
        albedo_list = []
        lvis_hit_list = []
        alpha_list = []
        import tqdm
        for i in tqdm.tqdm(range(0, n_pixel, chunk)):
            idx_chunk = idx
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            img_chunk = img[:, i:i+chunk, :]
            occu, depth, weight = self.get_depth(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk, feature_volume, sp_input, batch)
            surf_chunk = ray_o_chunk + ray_d_chunk * depth[:, None]
            new_batch = batch
            occu = torch.nn.functional.threshold(occu, 0.5, 0)
            alpha = torch.clip(occu, min=0., max=1.)

            with torch.no_grad():
                output, gt, _, _ = self.net((idx_chunk, None, ray_o_chunk, ray_d_chunk, img_chunk, None, surf_chunk, None, None, new_batch), mode='val', relight_olat=True, relight_probes=True)
            normal_pred = output['normal'].detach()
            lvis_pred = output['lvis'].detach()
            albedo_pred = output['albedo'].detach()
            brdf_pred = output['brdf'].detach()
            img_pred = output['rgb'].detach()
            relit_pred = output['rgb_olat'].detach()
            hdr_relit_pred = output['rgb_probes'].detach()
            normal_list.append(normal_pred)
            lvis_hit_list.append(lvis_pred)
            albedo_list.append(albedo_pred)
            relit_list.append(relit_pred)
            hdr_relit_list.append(hdr_relit_pred)
            rgb_list.append(img_pred)
            brdf_list.append(brdf_pred)
            alpha_list.append(alpha)


        brdf_pred = torch.cat(brdf_list, dim=0)
        albedo_pred = torch.cat(albedo_list, dim=0)
        normal_pred = torch.cat(normal_list, dim=0)#[chunk, 3]
        rgb_pred = torch.cat(rgb_list, dim=0)
        relit_pred = torch.cat(relit_list, dim=0)
        hdr_relit_pred = torch.cat(hdr_relit_list, dim=0)
        lvis_pred = torch.cat(lvis_hit_list, dim=0)
        alpha = torch.cat(alpha_list, dim=0)
        ret = {}
        ret['normal_pred'] = normal_pred
        ret['lvis_pred'] = lvis_pred
        ret['mask_at_box'] = mask_at_box
        ret['alpha_map'] = alpha.reshape(-1)
        ret['img'] = img # before relight
        ret['albedo_pred'] = albedo_pred
        ret['img_pred'] = rgb_pred
        ret['img_gt'] = img_gt
        ret['relit'] = relit_pred
        ret['brdf_pred'] = brdf_pred
        ret['hdr_relit'] = hdr_relit_pred
        return ret

    def relight_npose_render(self, batch):
        mask_at_box = batch['mask_at_box']

        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']

        with torch.no_grad():
            sp_input = self.prepare_sp_input(batch)
            feature_volume = self.net.net['neuralbody'].encode_sparse_voxels(sp_input)
            batch.update({
                'sp_input': sp_input,
                'feature_volume': feature_volume,
            })

        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 4096
        rgb_list = []
        brdf_list = []
        normal_list = []
        relit_list = []
        hdr_relit_list = []
        albedo_list = []
        lvis_hit_list = []
        alpha_list = []
        import tqdm
        for i in tqdm.tqdm(range(0, n_pixel, chunk)):
            idx_chunk = -1
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            img_chunk = None

            occu, depth, weight = self.get_depth(ray_o_chunk, ray_d_chunk, near_chunk, far_chunk, feature_volume, sp_input, batch)
            surf_chunk = ray_o_chunk + ray_d_chunk * depth[:, None]
            new_batch = batch
            occu = torch.nn.functional.threshold(occu, 0.5, 0)
            alpha = torch.clip(occu, min=0., max=1.)

            with torch.no_grad():
                output, gt, _, _ = self.net((idx_chunk, None, ray_o_chunk, ray_d_chunk, img_chunk, None, surf_chunk, None, None, new_batch), mode='val', relight_olat=True, relight_probes=True)
            normal_pred = output['normal'].detach()
            lvis_pred = output['lvis'].detach()
            albedo_pred = output['albedo'].detach()
            brdf_pred = output['brdf'].detach()
            img_pred = output['rgb'].detach()
            relit_pred = output['rgb_olat'].detach()
            hdr_relit_pred = output['rgb_probes'].detach()
            normal_list.append(normal_pred)
            lvis_hit_list.append(lvis_pred)
            albedo_list.append(albedo_pred)
            relit_list.append(relit_pred)
            hdr_relit_list.append(hdr_relit_pred)
            rgb_list.append(img_pred)
            brdf_list.append(brdf_pred)
            alpha_list.append(alpha)


        brdf_pred = torch.cat(brdf_list, dim=0)
        albedo_pred = torch.cat(albedo_list, dim=0)
        normal_pred = torch.cat(normal_list, dim=0)#[chunk, 3]
        rgb_pred = torch.cat(rgb_list, dim=0)
        relit_pred = torch.cat(relit_list, dim=0)
        hdr_relit_pred = torch.cat(hdr_relit_list, dim=0)
        lvis_pred = torch.cat(lvis_hit_list, dim=0)
        alpha = torch.cat(alpha_list, dim=0)
        ret = {}
        ret['normal_pred'] = normal_pred
        ret['lvis_pred'] = lvis_pred
        ret['mask_at_box'] = mask_at_box
        ret['alpha_map'] = alpha.reshape(-1)
        ret['albedo_pred'] = albedo_pred
        ret['img_pred'] = rgb_pred
        ret['relit'] = relit_pred
        ret['brdf_pred'] = brdf_pred
        ret['hdr_relit'] = hdr_relit_pred
        return ret
