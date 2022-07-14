import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import os.path as osp
import json
from .merl import MERL
from .renderer import SphereRenderer
from . import utils

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.mlp_chunk = args.mlp_chunk
        self.args = args
        self.embedder = self._init_embedder()
        self.net = self._init_net()

        data_dir = args.data_root
        train_npz = [f for f in os.listdir(data_dir) if f.startswith('train')]
        self.brdf_names = [
            osp.basename(x)[len('train_'):-len('.npz')] for x in train_npz]
        
        z_dim = args.z_dim
        z_gauss_mean = args.z_gauss_mean
        z_gauss_std = args.z_gauss_std
        normalize_z = args.normalize_z
        n_brdfs = len(self.brdf_names)
        self.latent_code = LatentCode(n_brdfs, z_dim, z_gauss_mean, z_gauss_std, normalize_z)#FIXME: maybe a passin params to facilitate optimizer

    def _init_embedder(self):
        n_freqs = self.args.n_freqs
        kwargs = {
            'input_dims': 3,
            'include_input': True,
            'max_freq_log2': n_freqs - 1,
            'num_freqs': n_freqs,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos]
        }
        embedder_rusink = Embedder(**kwargs)
        embedder = {'rusink': embedder_rusink}
        return embedder


    def _init_net(self):
        mlp_width = self.args.brdf_mlp_width
        mlp_depth = self.args.brdf_mlp_depth
        mlp_skip_at = self.args.brdf_mlp_skip_at
        net = nn.ModuleDict()
        net['brdf_mlp'] = MLP(
            18, [mlp_width] * mlp_depth, act=['relu'] * mlp_depth,
            skip_at=[mlp_skip_at])
        net['brdf_out'] = MLP(mlp_width, [1], act=['softplus']) # > 0
        return net

    def forward(self, batch, mode='train'):
        id_, i, envmap_h, ims, spp, rusink, refl = batch
        i =  i.reshape(-1)
        envmap_h = envmap_h.reshape(-1)
        ims = ims.reshape(-1)
        spp = spp.reshape(-1)
        rusink = rusink.reshape(-1, 3)
        refl = refl.reshape(-1, 1)
        if mode == 'test' and i[0] == -1:
            # print(id_)
            # print(i)
            # Novel identities -- need interpolation
            i_w1_mat1_w2_mat2 = id_[0]
            _, w1, mat1, w2, mat2 = i_w1_mat1_w2_mat2.split('_')
            w1, w2 = float(w1), float(w2)
            i1, i2 = self.brdf_names.index(mat1), self.brdf_names.index(mat2)
            z = self.latent_code.interp(w1, torch.Tensor([i1]).long().cuda(), w2, torch.Tensor([i2]).long().cuda())
            z = z.repeat(1, rusink.shape[0], 1).reshape(-1, 3)
        else:
            z = self.latent_code(i)
        brdf, brdf_reci = self._eval_brdf_at(z, rusink)
        # For loss computation
        pred = {'brdf': brdf, 'brdf_reci': brdf_reci}
        gt = {'brdf': refl}
        loss_kwargs = {}
        # To visualize
        to_vis = {
            'id': id_, 'i': i, 'z': z, 'gt_brdf': refl,
            'envmap_h': envmap_h, 'ims': ims, 'spp': spp}
        for k, v in pred.items():
            to_vis[k] = v
        return pred, gt, loss_kwargs, to_vis
    
    def _eval_brdf_at(self, z, rusink):
        mlp_layers = self.net['brdf_mlp']
        out_layer = self.net['brdf_out']
        # Chunk by chunk to avoid OOM
        chunks, chunks_reci = [], []
        for i in range(0, rusink.shape[0], self.mlp_chunk):
            end_i = min(rusink.shape[0], i + self.mlp_chunk)
            z_chunk = z[i:end_i]
            z_chunk = z_chunk.reshape(-1, 3)
            rusink_chunk = rusink[i:end_i, :]
            rusink_embed = self.embedder['rusink'](rusink_chunk)
            z_rusink = torch.cat((z_chunk, rusink_embed), dim=1)
            chunk = out_layer(mlp_layers(z_rusink))
            chunks.append(chunk)
            # Reciprocity
            phid = rusink[i:end_i, :1]
            thetah_thetad = rusink[i:end_i, 1:]
            rusink_chunk = torch.cat((phid + np.pi, thetah_thetad), dim=1)
            rusink_embed = self.embedder['rusink'](rusink_chunk)
            z_rusink = torch.cat((z_chunk, rusink_embed), dim=1)
            chunk = out_layer(mlp_layers(z_rusink))
            chunks_reci.append(chunk)
        brdf = torch.cat(chunks, dim=0)
        brdf_reci = torch.cat(chunks_reci, dim=0)
        return brdf, brdf_reci # (n_rusink, 1)
    
    def compute_loss(self, pred, gt, **kwargs):
        loss_transform = torch.log
        loss = 0
        weight  = 1.
        loss_func = nn.MSELoss()
        loss += weight * loss_func(loss_transform(gt['brdf']), loss_transform(pred['brdf']))
        loss += weight * loss_func(loss_transform(gt['brdf']), loss_transform(pred['brdf_reci']))
        return loss

    def vis_batch(self, data_dict, outdir, mode='train', dump_raw_to=None, n_vis=64):
        # Shortcircuit if training
        if mode == 'train':
            return
        # Optionally dump raw to disk
        if dump_raw_to is not None:
            # ioutil.dump_dict_tensors(data_dict, dump_raw_to)
            pass
        # "Visualize" metadata
        id_ = data_dict['id'][0]
        metadata_out = osp.join(outdir, 'metadata.json')
        metadata = {'id': id_}
        write_json(metadata, metadata_out)
        # Visualize the latent codes
        z = data_dict['z'][0, :].cpu().numpy()
        z_png = osp.join(outdir, 'z.png')
        plot = utils.plot.Plot(outpath=z_png)
        plot.bar(z)
        # Visualize the BRDF values
        pred = data_dict['brdf'].cpu().numpy()
        pred_reci = data_dict['brdf_reci'].cpu().numpy()
        brdf_val = np.hstack((pred_reci, pred))
        labels = ['Pred. (reci.)', 'Pred.']
        if mode == 'vali':
            gt = data_dict['gt_brdf'].cpu().numpy()
            brdf_val = np.hstack((brdf_val, gt))
            labels.append('GT')
        brdf_val = brdf_val[::int(brdf_val.shape[0] / n_vis), :] # just a subset
        brdf_val = np.log10(brdf_val) # log scale
        brdf_png = osp.join(outdir, 'log10_brdf.png')
        plot = utils.plot.Plot(labels=labels, outpath=brdf_png)
        plot.bar(brdf_val)
        if mode == 'vali':
            return
        # If testing, continue to visualize characteristic slice
        merl = MERL()
        envmap_h = data_dict['envmap_h'][0].cpu().numpy()
        ims = data_dict['ims'][0].cpu().numpy()
        spp = data_dict['spp'][0].cpu().numpy()
        renderer = SphereRenderer(
            'point', outdir, envmap_h=envmap_h, envmap_inten=40, ims=ims,
            spp=spp)
        cslice_out = osp.join(outdir, 'cslice.png')
        cslice_shape = merl.cube_rusink.shape[1:]
        cslice_end_i = np.prod(cslice_shape[:2])
        pred_cslice = pred[:cslice_end_i, :] # first 90x90 are for char. slices
        cslice = pred_cslice.reshape(cslice_shape[:2])
        cslice_img = merl.characteristic_slice_as_img(cslice)
        cv2.imwrite(cslice_out, cslice_img)
        # xm.io.img.write_img(cslice_img, cslice_out)
        # ... and render the predicted BRDF
        render_out = osp.join(outdir, 'render.png')
        pred_render = pred[cslice_end_i:, :] # remaining are for rendering
        brdf = np.zeros_like(renderer.lcontrib)
        brdf[renderer.lvis.astype(bool)] = pred_render
        render = renderer.render(brdf)
        render = np.clip(render, 0, 1)
        cv2.imwrite(render_out, render*255)
        # xm.io.img.write_arr(render, render_out, clip=True)

def write_json(data, path):
    out_dir = osp.dirname(path)
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    with open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)

class MLP(nn.Module):
    def __init__(self, input_dim, widths, act=None, skip_at=None):
        super(MLP, self).__init__()
        depth = len(widths)
        self.input_dim = input_dim

        if act is None:
            act = [None] * depth
        assert len(act) == depth
        self.layers = nn.ModuleList()
        self.activ = None
        prev_w = self.input_dim
        i = 0
        for w, a in zip(widths, act):
            if isinstance(a, str):
                if a == 'relu':
                    self.activ = nn.ReLU()
                elif a == 'softplus':
                    self.activ = nn.Softplus()
                else:
                    raise NotImplementedError
            layer = nn.Linear(prev_w, w)
            prev_w = w
            if skip_at and i in skip_at:
                prev_w += input_dim
            self.layers.append(layer)
            i += 1
        self.skip_at = skip_at

    def forward(self, x):
        x_ = x + 0
        for i, layer in enumerate(self.layers):
            # print(i)
            # print(x_.shape)
            y = self.activ(layer(x_))
            if self.skip_at and i in self.skip_at:
                y = torch.cat((y, x), dim=-1)
            x_ = y
            # print(y.shape)
        return y

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class LatentCode(nn.Module):
    def __init__(self, n_iden, dim, mean=0., std=1., normalize=False):
        super(LatentCode, self).__init__()
        self._z = torch.randn(n_iden, dim)*std+mean
        self._z = nn.parameter.Parameter(self._z)
        self.normalize = normalize

    @property
    def z(self):
        """The exposed interface for retrieving the current latent codes.
        """
        if self.normalize:
            return nn.functional.normalize(self._z, p=2, dim=1, eps=1e-6)
        return self._z

    def forward(self, ind):
        # ind = torch.Tensor(ind)
        if len(ind.shape) == 0:
            ind = ind.reshape((1,))
        # print(ind)
        z = self.z[ind[:,None]]
        return z

    def interp(self, w1, i1, w2, i2):
        z1, z2 = self(i1),  self(i2)
        if self.normalize:
            assert w1 + w2 == 1., \
                "When latent codes are normalized, use weights that sum to 1"
            z = slerp(z1, z2, w2)
        else:
            z = w1 * z1 + w2 * z2
        return z

def slerp(p0, p1, t):
    assert p0.ndim == p1.ndim == 2, "Vectors must be 2D"

    if p0.shape[0] == 1:
        cos_omega = p0 @ torch.transpose(p1)
    elif p0.shape[1] == 1:
        cos_omega = torch.transpose(p0) @ p1
    else:
        raise ValueError("Vectors should have one singleton dimension")

    omega = torch.acos(torch.clip(cos_omega, min=-1., max=1.))

    z0 = p0 * torch.sin((1 - t) * omega) / torch.sin(omega)
    z1 = p1 * torch.sin(t * omega) / torch.sin(omega)

    z = z0 + z1
    return z