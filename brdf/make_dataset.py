from ast import parse
import os
import os.path as osp
from numpy.lib.polynomial import polyint
from tqdm import tqdm
import numpy as np
from io import BytesIO
import cv2
from argparse import ArgumentParser

from renderer import SphereRenderer
from merl import MERL
import utils as utils


def save_npz(dict_, path):
    """The extra hassle is for Google infra.
    """
    with open(path, 'wb') as h:
        io_buffer = BytesIO()
        np.savez(io_buffer, **dict_)
        h.write(io_buffer.getvalue())

def main(args):
    if not osp.exists(args.outdir):
        os.makedirs(args.outdir)

    brdf = MERL()

    renderer = SphereRenderer(args.envmap_path, args.outdir, envmap_inten=args.envmap_inten, envmap_h=args.envmap_h, ims=args.ims, spp=args.spp)

    # First 90x90 Rusink. are for the characteristic slice
    cslice_rusink = brdf.get_characterstic_slice_rusink()
    cslice_rusink = np.reshape(cslice_rusink, (-1, 3))

    # Next are for rendering
    render_rusink = brdf.dir2rusink(renderer.ldir, renderer.vdir)
    render_rusink = render_rusink[renderer.lvis.astype(bool)]

    qrusink = np.vstack((cslice_rusink, render_rusink))

    data = {
        'envmap_h': args.envmap_h, 'ims': args.ims, 'spp': args.spp,
        'rusink': qrusink.astype(np.float32)}

    out_path = osp.join(args.outdir, 'test.npz')
    save_npz(data, out_path)

    # ------ Training & Validation

    brdf_paths = sorted(os.listdir(args.indir))
    # print(brdf_paths)
    for i, path in enumerate(tqdm(brdf_paths, desc="Training & Validation")):
        path = osp.join(args.indir, path)
        brdf = MERL(path=path)

        rusink = brdf.tbl[:, :3]
        refl = brdf.tbl[:, 3:]
        refl = utils.camera.rgb2lum(refl)
        refl = refl[:, None]

        # Training-validation split
        n = brdf.tbl.shape[0]
        take_every = int(1 / args.vali_frac)
        ind = np.arange(0, n)
        vali_ind = np.arange(0, n, take_every, dtype=int)
        train_ind = np.array([x for x in ind if x not in vali_ind])
        train_rusink = rusink[train_ind, :]
        train_refl = refl[train_ind, :]
        vali_rusink = rusink[vali_ind, :]
        vali_refl = refl[vali_ind, :]

        train_data = {
            'i': i, 'name': brdf.name,
            'envmap_h': args.envmap_h, 'ims': args.ims, 'spp': args.spp,
            'rusink': train_rusink.astype(np.float32),
            'refl': train_refl.astype(np.float32)}
        vali_data = {
            'i': i, 'name': brdf.name,
            'envmap_h': args.envmap_h, 'ims': args.ims, 'spp': args.spp,
            'rusink': vali_rusink.astype(np.float32),
            'refl': vali_refl.astype(np.float32)}

        # Dump to disk
        out_path = osp.join(args.outdir, 'train_%s.npz' % brdf.name)
        save_npz(train_data, out_path)
        out_path = osp.join(args.outdir, 'vali_%s.npz' % brdf.name)
        save_npz(vali_data, out_path)

        # Visualize
        vis_dir = osp.join(args.outdir, 'vis')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)
        for achro in (False, True):
            # Characteristic slice
            cslice = brdf.get_characterstic_slice()
            if achro:
                cslice = utils.camera.rgb2lum(cslice)
                cslice = np.tile(cslice[:, :, None], (1, 1, 3))
            cslice_img = brdf.characteristic_slice_as_img(
                cslice, clip_percentile=args.slice_percentile)
            folder_name = 'cslice'
            if achro:
                folder_name += '_achromatic'
            out_png = osp.join(vis_dir, folder_name, brdf.name + '.png')
            cv2.imwrite(out_png, cslice_img)
            # Render with this BRDF
            qrusink = brdf.dir2rusink(renderer.ldir, renderer.vdir)
            lvis = renderer.lvis.astype(bool)
            qrusink_flat = qrusink[lvis]
            rgb_flat = brdf.query(qrusink_flat)
            rgb = np.zeros_like(renderer.lcontrib)
            rgb[lvis] = rgb_flat
            if achro:
                rgb = utils.camera.rgb2lum(rgb)
                rgb = np.tile(rgb[:, :, :, None], (1, 1, 1, 3))
            render = renderer.render(rgb)
            folder_name = 'render'
            if achro:
                folder_name += '_achromatic'
            out_png = osp.join(vis_dir, folder_name, brdf.name + '.png')
            render = np.clip(render, 0., 1.)
            render = (render*np.iinfo('uint8').max).astype('uint8')
            cv2.imwrite(out_png, render)
            # xm.io.img.write_arr(render, out_png, clip=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--vali_frac', type=int, default=0.01)
    parser.add_argument('--envmap_path', type=str, default='point')
    parser.add_argument('--envmap_h', type=int, default=16)
    parser.add_argument('--envmap_inten', type=int, default=40)
    parser.add_argument('--slice_percentile', type=int, default=80)
    parser.add_argument('--ims', type=int, default=128)
    parser.add_argument('--spp', type=int, default=1)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()
    main(args)