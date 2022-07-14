import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
import torch.nn.functional as F
from lib.networks.relight_microfacet import Network


class NetworkWrapper(nn.Module):
    def __init__(self, net: Network):
        super(NetworkWrapper, self).__init__()
        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net.net['neuralbody'])

    def forward(self, batch):
        ret = self.renderer.relighte2e_render(batch)

        scalar_stats = {}
        loss = 0

        alpha = ret['alpha_map']
        mask = batch['mask_at_box']
        img_gt = batch['rgb']
        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        surf = ret['surf']
        normal = ret['normal']
        lvis = ret['lvis_hit']

        batch.update({
            'sp_input': ret['sp_input'],
            'feature_volume': ret['feature_volume']
        })
        pred, gt, loss_kwargs, to_vis = self.net((-1, None, ray_o, ray_d, img_gt, alpha, surf, normal, lvis, batch))
        loss_sum, normal_loss, lvis_loss, normal_smooth_loss, lvis_smooth_loss, albedo_smooth_loss, brdf_smooth_loss, rgb_loss, albedo_entropy = self.net.compute_loss(pred, gt, **loss_kwargs)
        loss = loss + loss_sum
        if cfg.mask_loss > 0:
            if 'mmsk' in batch:
                mask_loss = F.binary_cross_entropy(alpha.clip(1e-3, 1.0-1e-3), batch['mmsk'])
                loss += cfg.mask_loss * mask_loss
                scalar_stats.update({'mask_loss': mask_loss})


        scalar_stats.update({'loss': loss, 'normal_loss': normal_loss, 'lvis_loss': lvis_loss, 'normal_smooth_loss': normal_smooth_loss, 'lvis_smooth_loss': lvis_smooth_loss, 'albedo_smooth_loss': albedo_smooth_loss, 'brdf_smooth_loss': brdf_smooth_loss, 'rgb_loss': rgb_loss, 'albedo_entropy': albedo_entropy})
        image_stats = {}

        return pred, loss, scalar_stats, image_stats
