from math import atanh
import os
import os.path as osp
import torch
import torch.utils.data as data
import numpy as np

def load_np(np_f):
    if np_f.endswith('.npy'):
        with open(np_f, 'rb') as h:
            data = np.load(h)
        return data

    # .npz
    with open(np_f, 'rb') as h:
        data = np.load(h, allow_pickle=True)
        data = dict(data)
    return data

class BRDFDataset(data.Dataset):
    def __init__(self, args, mode, seed=0, n_iden=20, n_between=11):
        super(BRDFDataset, self).__init__()
        assert mode in ('train', 'vali', 'test'), (
            "Accepted dataset modes: 'train', 'vali', 'test', but input is %s"
        ) % mode
        self.args = args
        self.mode = mode
        self.bs = args.n_rays_per_step #n_rays_per_step

        root = args.data_root # path to saved npz
        all_paths = os.listdir(root)
        train_paths = []
        vali_paths = []
        test_paths = []
        for filename in all_paths:
            if filename.startswith('train'):
                filename = osp.join(args.data_root, filename)
                train_paths.append(filename)
            elif filename.startswith('vali'):
                filename = osp.join(args.data_root, filename)
                vali_paths.append(filename)
            elif filename.startswith('test'):
                filename = osp.join(args.data_root, filename)
                test_paths.append(filename)
        assert len(test_paths) == 1, (
            "There should be a single set of test coordinates, shared by "
            "all identities")
        self.brdf_names = [
            osp.basename(x)[len('train_'):-len('.npz')] for x in train_paths]
        
        self.test_data = load_np(test_paths[0])
        test_paths = []
        # First 100: novel Rusink., but seen identities
        test_paths += self.brdf_names
        # Next: novel Rusink. and interpolated identities
        np.random.seed(seed) # fix random seed
        mats = np.random.choice(self.brdf_names, n_iden, replace=False)
        i = 0
        for mat_i in range(n_iden - 1):
            mat1, mat2 = mats[mat_i], mats[mat_i + 1]
            for a in np.linspace(1, 0, n_between, endpoint=True):
                b = 1 - a
                id_ = f'{i:06d}_{a:f}_{mat1}_{b:f}_{mat2}'
                test_paths.append(id_)
                i += 1
        np.random.seed() # restore random seed
        self.paths = {
            'train': train_paths, 'vali': vali_paths, 'test': test_paths}
        self.files = sorted(self.paths[self.mode])
        assert self.files, "No file to process into a dataset"

    def _load_data(self, path):
        if self.mode == 'test':
            id_ = path
            data = self.test_data
        else:
            data = load_np(path)
        envmap_h = data['envmap_h'][()]
        ims = data['ims'][()]
        spp = data['spp'][()]
        rusink = data['rusink']
        if self.mode == 'test':
            i = self.brdf_names.index(id_) if id_ in self.brdf_names else -1
            refl = np.zeros(
                (rusink.shape[0], 1), dtype=rusink.dtype) # placeholder
        else:
            id_ = data['name'][()]
            i = data['i'][()]
            refl = data['refl']
        return id_, np.array([i]), np.array([envmap_h]), np.array([ims]), np.array([spp]), rusink.reshape(-1, 3), refl.reshape(-1, 1)

    def _sample_entries(self, tbls):
        n_rows = None
        for tbl in tbls:
            if n_rows is None:
                n_rows = tbl.shape[0]
            else:
                assert tbl.shape[0] == n_rows
        if self.mode in ('vali', 'test'):
            return tbls
        
        selected_idx = np.random.uniform(low=0, high=tbls[0].shape[0], size=(self.bs,)).astype('int32')
        selected_idx = selected_idx[:, None]
        ret = []
        for tbl in tbls:
            tbl_rows = tbl[selected_idx,:]
            ret.append(tbl_rows)
        return ret

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        id_, i, envmap_h, ims, spp, rusink, refl = self._load_data(path)

        rusink, refl = self._sample_entries((rusink, refl))
        n = rusink.shape[0]

        # id_ = np.tile(id_[None, :], (n,))
        if self.mode == 'test' and i[0] == -1:
            return id_, i, envmap_h, ims, spp, rusink, refl
        else:
            id_ = [id_] * n
            i = np.tile(i[None, :], (n,)).reshape(-1)
            envmap_h = np.tile(envmap_h[None, :], (n,)).reshape(-1)
            ims = np.tile(ims[None, :], (n,)).reshape(-1)
            spp = np.tile(spp[None, :], (n,)).reshape(-1)
            return id_, i, envmap_h, ims, spp, rusink, refl
