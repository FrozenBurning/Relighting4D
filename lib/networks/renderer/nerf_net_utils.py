import torch.nn.functional as F
import torch
from lib.config import cfg


def density2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) *
                                                                 dists)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists,
         torch.Tensor([0]).expand(dists[..., :1].shape).to(dists)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 0].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 0] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return None, disp_map, acc_map, weights, depth_map

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) *
                                                                 dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [dists,
         torch.Tensor([0]).expand(dists[..., :1].shape).to(dists)],
        -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones((alpha.shape[0], 1)).to(alpha), 1. - alpha + 1e-10],
            -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(depth_map),
                              depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map
