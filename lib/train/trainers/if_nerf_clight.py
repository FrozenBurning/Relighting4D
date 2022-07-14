import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import if_clight_renderer
from lib.train import make_optimizer
import torch.nn.functional as F

class CorrespondenceLoss():
    def __init__(self):
        self.avg = None
        self.n = 0
        self.crit = torch.nn.functional.smooth_l1_loss

    def __call__(self, vert_density):
        self.n += 1
        if self.avg is not None:
            loss = self.crit(self.avg, vert_density)
            self.avg = (self.avg*(self.n-1) + vert_density)/self.n
        else:
            # init
            self.avg = vert_density
            loss = self.crit(self.avg, vert_density)
        self.avg = self.avg.detach()
        return loss

class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = if_clight_renderer.Renderer(self.net)

        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)
        self.acc_crit = torch.nn.functional.smooth_l1_loss
        self.corrloss = CorrespondenceLoss()

    def forward(self, batch):
        ret = self.renderer.render(batch)

        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        if 'male' in cfg.exp_name:
            loss += img_loss * 10
        else:
            loss += img_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        if cfg.mask_loss > 0:
            if 'mmsk' in batch:
                mask_loss = F.binary_cross_entropy(ret['acc_map'].clip(1e-3, 1-1e-3), batch['mmsk'])
                loss += cfg.mask_loss * mask_loss
                scalar_stats.update({'mask_loss': mask_loss})
        
        if 'wxyz' in batch:
            corr_loss = self.corrloss(ret['corr_density'])
            loss += corr_loss
            scalar_stats.update({'corr_loss': corr_loss})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
