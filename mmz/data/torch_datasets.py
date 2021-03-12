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

class BaseDataset(data.Dataset):
    env_key = None
    @staticmethod
    def data_loader_from_dataset(dset, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, **kwargs):
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


    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, **kwargs):
        dl = self.data_loader_from_dataset(self, batch_size=batch_size,
                                      num_workers=num_workers,
                                      batches_per_epoch=batches_per_epoch,
                                      random_sample=random_sample,
                                      shuffle=shuffle, **kwargs)
        return dl

class FileDirDataset(BaseDataset):
    def __init__(self, dataroot, load_func=None, filter_func=None, transform=None):
        self.load_func = self.image_loader if load_func is None else load_func
        self.transform = transform
        self.filter_func = filter_func

        self.file_path_d = self.create_file_map(dataroot, filter_func=self.filter_func)
        import pandas as pd
        # Series with index as the file name (key of dict) and full path as value
        self.path_df = pd.Series(self.file_path_d, name='full_path').to_frame()
        self.label_df = pd.DataFrame(index=self.path_df.index)

        self.all_paths = self.path_df.full_path.values
        self.all_labels = None

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, item):
        p = self.all_paths[item]
        obj = self.load_func(p)
        if self.transform is not None:
            obj = self.transform(obj)

        if self.label_df.shape[1] > 0:
            obj = (obj, self.all_labels[item])

        return obj


    def join_labels(self, name, label_s):
        self.label_df[name] = label_s
        self.all_labels = self.label_df.values
        return self

    @staticmethod
    def image_loader(path):
        from PIL import Image
        img = Image.open(path)
        #with open(path, 'rb') as f:
        #    img = Image.open(f).load()
        return img

    @classmethod
    def create_file_map(cls, dataroot, filter_func=None):
        import os
        fname_path_d = dict()
        for root, dirs, files in os.walk(dataroot):
            for name in files:
                #print(os.path.join(root, name))
                full_path = os.path.join(root, name)
                if filter_func is None or filter_func(full_path):
                    fname_path_d[name] = full_path
        return fname_path_d


def make_fashion_mnist(batch_size=128, num_workers=4, image_size=28, dataroot='~/datasets'):
    # from torch.utils.data import dataset as dset
    # from torchvision import dataset
    import torchvision.datasets as dset
    import torchvision.transforms as transforms
    # Root directory for dataset
    #dataroot = "~/datasets"

    #image_size = 28
    #batch_size = 128
    #workers = 4

    fashion_dataset = dset.FashionMNIST(dataroot, train=True,
                                        transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            # transforms.Normalize(0.5, 0.5),
                                            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]), target_transform=None, download=True)
    dl = BaseDataset.data_loader_from_dataset(fashion_dataset, batch_size=batch_size,
                                              num_workers=num_workers)
    return fashion_dataset, dl

    # Create the dataloader
    #dl = torch.utils.data.DataLoader(fashion_dataset, batch_size=batch_size,
    #                                 sampler=torch.utils.data.RandomSampler(fashion_dataset,
    #                                                                        replacement=True,
    #                                                                        num_samples=100 * batch_size),
    #                                 shuffle=False, num_workers=workers)

    # Decide which device we want to run on
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        self.fid_to_path = {split(f)[-1]: f
                            for f in tqdm(self.files, desc="Getting files")}


        if self.num_samples is None:
            self.train_ixes = self.attr_df.index.tolist()
        else:
            self.train_ixes = np.random.choice(self.attr_df.index, self.num_samples,
                                               replace=True)

        if self.transforms is None:
            self.transforms = trfs.Compose([
                                            trfs.Resize(64),
                                            trfs.CenterCrop(64),
                                            trfs.ToTensor(),
                                            trfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
        return len(self.train_ixes)

