import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
from lib.train import make_optimizer
from lib.networks.shape import Network


class NetworkWrapper(nn.Module):
    def __init__(self, net: Network):
        super(NetworkWrapper, self).__init__()
        self.net = net

    def forward(self, batch):

        scalar_stats = {}
        loss = 0

        idx = batch['idx']
        mask = batch['mask_at_box']
        img = batch['img']
        surf = batch['surf']
        normal = batch['normal']
        alpha = batch['alpha']
        lvis = batch['lvis']
        light_map = batch['light_map']
        pred, gt, loss_kwargs, to_vis = self.net((idx, None, alpha, surf, normal, lvis, light_map))
        loss_sum, normal_loss, lvis_loss = self.net.compute_loss(pred, gt, **loss_kwargs)
        loss = loss + loss_sum


        scalar_stats.update({'loss': loss, 'normal_loss': normal_loss, 'lvis_loss': lvis_loss})
        image_stats = {}

        return pred, loss, scalar_stats, image_stats
