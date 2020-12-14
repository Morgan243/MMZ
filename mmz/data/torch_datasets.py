import attr
import torch
import pandas as pd
import itertools

from torch.utils import data
from torchvision import transforms as trfs
from PIL import Image
from tqdm.auto import tqdm

from glob import glob
from os.path import split, join
import numpy as np

@attr.attrs
class NDGaussian(data.Dataset):
    n = attr.ib()
    centers = attr.ib()
    cov_mat = attr.ib(None)
    dims = attr.ib(None)

    def __attrs_post_init__(self):
        X_Y = self.make_multi_n_normal(centers=self.centers,
                                       cov_mat=self.cov_mat, dims=self.dims,
                                       size=self.n).astype('float32')
        self.X, self.Y = X_Y[:, :-1], X_Y[:, -1:]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def __len__(self):
        return self.X.shape[0]

    @staticmethod
    def make_multi_n_normal(centers, dims=None, cov_mat=None, size=100, include_labels=True):
        assert len(set(len(c) for c in centers)) == 1
        N = (dims if dims is not None else len(centers[0]))
        cov_mat = np.eye(N) if cov_mat is None else cov_mat
        size_per_center = size // N

        arrs = [np.concatenate([np.random.multivariate_normal(
            mean=np.concatenate([dim_means, np.zeros(N - len(dim_means))]),
            cov=cov_mat,
            size=size_per_center),
            np.ones((size_per_center, 1)) * i
        ], axis=1)
            for i, dim_means in enumerate(centers)]

        syn_arr = np.concatenate(arrs)
        if not include_labels:
            syn_arr = syn_arr[:, :-1]
        return syn_arr

    @staticmethod
    def centers_as_nd_square(n, mu_sep, dims=2, centered=True):
        """
        Returns an array of shape (n*n, 2) points making a n squared frid in a 2d plane
        """
        mus = np.array(list(itertools.product(*[np.arange(0, n * mu_sep, mu_sep) for _ in range(dims)])))
        if centered:
            mus = mus - mus.max() / 2

        return mus

    @staticmethod
    def centers_as_2d_circle(n, radius):
        i = np.arange(0, n)
        x = radius * np.cos(2 * np.pi / n * i)
        y = radius * np.sin(2 * np.pi / n * i)
        mus = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], -1)
        return mus


@attr.attrs
class CelebA(data.Dataset):
    dataroot = attr.ib()
    attr_path = attr.ib()
    labels = attr.ib(['Chubby'])
    transforms = attr.ib(None)
    num_samples = attr.ib(5000)


    def __attrs_post_init__(self):
        self.prep_attributes()
        self.files = glob(join(self.dataroot, "img_align_celeba", "*"))
        #self.num_files = len(self.files)
        #self.num_images = self.num_files
        self.num_images = self.num_files = self.num_samples

        assert self.num_files <= len(self.attr_df)

        self.fid_to_path = {split(f)[-1]: f
                            for f in tqdm(self.files, desc="Getting files")}


        self.train_ixes = np.random.choice(self.attr_df.index, self.num_samples,
                                           replace=True)
        self.test_ixes = np.random.choice([i for i in self.attr_df.index
                                           if i not in self.train_ixes],
                                        1000, replace=False)

        self.transforms = trfs.Compose([
                                        trfs.Resize(64),
                                        trfs.CenterCrop(64),
                                        trfs.ToTensor(),
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])

    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, **kwargs):
        dset = self
        if random_sample:
            if batches_per_epoch is None:
                batches_per_epoch = len(dset) // batch_size

            dataloader = data.DataLoader(dset, batch_size=batch_size,
                                          sampler=data.RandomSampler(dset,
                                                                      replacement=True,
                                                                      num_samples=batches_per_epoch * batch_size),
                                          shuffle=shuffle, num_workers=num_workers,
                                          **kwargs)
        else:
            dataloader = data.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers,
                                          **kwargs)
        return dataloader

    def prep_attributes(self):
        self.attr_df = pd.read_csv(self.attr_path)
        self.attr_df.replace(-1, 0, inplace=True)
        self.attr_df = self.attr_df.set_index('image_id')

    def __getitem__(self, item):
        ix = self.train_ixes[item]
        path = self.fid_to_path[ix]
        image = Image.open(path)

        label_arr = self.attr_df.loc[ix][self.labels].values
        label_arr = torch.from_numpy(label_arr).float()
        return self.transforms(image), torch.FloatTensor(label_arr)

    def __len__(self):
        return self.num_images
