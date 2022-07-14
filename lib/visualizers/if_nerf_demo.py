import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from lib.config import cfg
import cv2
import os
from termcolor import colored
from io import BytesIO

def save_npz(dict_, path):
    """The extra hassle is for Google infra.
    """
    with open(path, 'wb') as h:
        io_buffer = BytesIO()
        np.savez(io_buffer, **dict_)
        h.write(io_buffer.getvalue())

class Visualizer:
    def __init__(self):
        data_dir = 'data/render/{}'.format(cfg.exp_name)
        print(colored('the results are saved at {}'.format(data_dir), 'yellow'))

    def alpha_blend(self, tensor1, alpha, tensor2=None):
        if tensor2 is None:
            tensor2 = np.zeros_like(tensor1)
        if len(np.shape(tensor1)) == 3 and len(np.shape(alpha)) == 2:
            alpha = np.reshape(alpha, tuple(np.shape(alpha)) + (1,))
            alpha = np.tile(alpha, (1, 1, np.shape(tensor1)[2]))
        return np.multiply(tensor1, alpha) + np.multiply(tensor2, 1. -alpha)

    def visualize_relight(self, output, batch):
        print("visualizing......")
        rgb_pred = output['img_pred'].detach().cpu().numpy()
        albedo = output['albedo_pred'].detach().cpu().numpy()
        relit = output['relit'].detach().cpu().numpy()
        hdr_relit = output['hdr_relit'].detach().cpu().numpy()
        alpha_map = output['alpha_map'].detach().cpu().numpy()
        normal_pred = output['normal_pred'].detach().cpu().numpy()
        lvis_pred = output['lvis_pred'].detach().cpu().numpy()
        brdf_pred = output['brdf_pred'].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        rgb_pred = self.alpha_blend(rgb_pred, alpha_map[:, None], np.zeros_like(rgb_pred))

        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]
        num_olat = relit.shape[-2]
        img_relit = np.zeros((H, W, num_olat, 3))
        relit = relit.reshape(-1, num_olat*3)
        relit = self.alpha_blend(relit, alpha_map[:, None], np.zeros_like(relit))
        relit = relit.reshape(-1, num_olat, 3)
        img_relit[mask_at_box] = relit
        img_relit = img_relit[..., [2, 1, 0]]

        num_hdr = hdr_relit.shape[1]
        img_hdr_relit = np.zeros((H, W, num_hdr, 3))
        hdr_relit = hdr_relit.reshape(-1, num_hdr*3)
        hdr_relit = self.alpha_blend(hdr_relit, alpha_map[:, None], np.zeros_like(hdr_relit))
        hdr_relit = hdr_relit.reshape(-1, num_hdr, 3)
        img_hdr_relit[mask_at_box] = hdr_relit
        img_hdr_relit = img_hdr_relit[..., [2, 1, 0]]

        output_albedo = np.zeros((H, W, 3))
        output_albedo_gamma = np.zeros((H, W, 3))

        albedo_gamma = albedo ** (1/ 2.2)
        albedo = self.alpha_blend(albedo, alpha_map[:, None], np.zeros_like(albedo))
        albedo_gamma = self.alpha_blend(albedo_gamma, alpha_map[:, None], np.zeros_like(albedo_gamma))
        output_albedo[mask_at_box] = albedo
        output_albedo = output_albedo[..., [2, 1, 0]]
        output_albedo_gamma[mask_at_box] = albedo_gamma
        output_albedo_gamma = output_albedo_gamma[..., [2, 1, 0]]

        output_alpha_map = np.zeros((H, W, 1))
        output_normal_map = np.zeros((H, W, 3))
        output_normal_map_pred = np.zeros((H, W, 3))
        output_lvis_map_pred = np.zeros((H, W, lvis_pred.shape[-1]))
        output_brdf_pred = np.ones((H, W, brdf_pred.shape[-1]))

        output_brdf_pred[mask_at_box] = brdf_pred
        output_brdf_pred[mask_at_box] = self.alpha_blend(output_brdf_pred[mask_at_box], alpha_map[:, None], np.array([1.]))

        output_lvis_map_pred[mask_at_box] = lvis_pred
        output_lvis_map_pred[mask_at_box] = self.alpha_blend(output_lvis_map_pred[mask_at_box], alpha_map[:, None])

        output_alpha_map[mask_at_box] = alpha_map[:, None]

        normal_map_bg = np.array([0., 0., 0.])
        normal_map = self.alpha_blend(normal_pred.reshape((-1,3)), alpha_map[:, None], normal_map_bg)
        normal_map = normal_map / np.linalg.norm(normal_map, ord=2, axis=-1)[:, None]
        normal_map = np.clip(normal_map, -1., 1.)
        output_normal_map_pred[mask_at_box] = (normal_map + 1.) /2
        output_normal_map_pred = output_normal_map_pred[..., [2, 1, 0]]

        img_root = 'data/render/{}/{}'.format(cfg.task,
            cfg.exp_name)
        os.system('mkdir -p {}'.format(img_root))
        index = batch['idx'].item()
        cv2.imwrite(os.path.join(img_root, '{:04d}_rgbpred.png'.format(index)), img_pred * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_albedo.png'.format(index)), output_albedo * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_albedo_gamma.png'.format(index)), output_albedo_gamma * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_normal_pred.png'.format(index)), output_normal_map_pred * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_alpha.png'.format(index)), output_alpha_map * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_lvis_pred.png'.format(index)), np.mean(output_lvis_map_pred,axis=2) * 255)

        for hdr_i in range(num_hdr):
            cv2.imwrite(os.path.join(img_root, '{:04d}_{:01d}_hdr_relit.png'.format(index, hdr_i)), img_hdr_relit[:, :, hdr_i, :]*255)

        vis_out = os.path.join(img_root, '{:04d}_lvis_frames'.format(index))
        os.system('mkdir -p {}'.format(vis_out))
        olats = [27, 91, 149, 200, 288, 333, 398, 488]
        for i in range(num_olat):
            cv2.imwrite(os.path.join(vis_out, '{:04d}_lvis_pred.png'.format(olats[i])), output_lvis_map_pred[:,:,olats[i]] * 255)
            cv2.imwrite(os.path.join(vis_out, '{:04d}_relit.png'.format(olats[i])), img_relit[:, :, i, :]* 255)


    def visualize(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        if cfg.white_bkgd:
            img_pred = img_pred + 1
        img_pred[mask_at_box] = rgb_pred
        img_pred = img_pred[..., [2, 1, 0]]

        depth_pred = np.zeros((H, W))
        depth_pred[mask_at_box] = output['disp_map'][0].detach().cpu().numpy()

        acc_pred = np.zeros((H, W))
        acc_pred[mask_at_box] = output['acc_map'][0].detach().cpu().numpy()

        img_root = 'data/render/{}/frame_{:04d}'.format(cfg.exp_name, batch['frame_index'].item())
        os.system('mkdir -p {}'.format(img_root))
        index = batch['view_index'].item()

        cv2.imwrite(os.path.join(img_root, '{:04d}.png'.format(index)),
                    img_pred * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_mask.png'.format(index)), mask_at_box * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_depth.png'.format(index)), depth_pred * 255)
        cv2.imwrite(os.path.join(img_root, '{:04d}_alpha.png'.format(index)), acc_pred * 255)