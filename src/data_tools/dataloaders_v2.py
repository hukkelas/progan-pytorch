import torch
import os
import src.utils
import glob
import numpy as np
from torchvision import transforms
from src.data_tools.data_utils import DataPrefetcher
import torchvision.datasets as datasets

def load_dataset(dataset, batch_size, imsize, full_validation, load_fraction=False):
    if dataset == "celeba-HQ":
        return _load_dataset("data/celeba-HQ", imsize, batch_size, full_validation, load_fraction)
    if dataset == "mnist":
        return DataPrefetcher(load_mnist(batch_size, imsize))
    raise AssertionError("Dataset was not found", dataset)


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, images, augment_data):
        self.augment_data = augment_data
        self.images = images
        self.imsize = self.images.shape[1]
        expected_imshape = (self.imsize, self.imsize, 3)
        assert self.images.shape[1:] == expected_imshape, "Shape was: {}. Expected: {}".format(
            self.images.shape[1:], expected_imshape)
        print("Dataset loaded. Number of samples:", self.images.shape)
        self.images = [transforms.functional.to_pil_image(im) for im in self.images]

    def __getitem__(self, index):
        im = self.images[index]
        if self.augment_data and np.random.rand() > 0.5:
            im = transforms.functional.hflip(im)
        im = np.asarray(im)
        return im

    def __len__(self):
        return len(self.images)

def fast_collate(imgs):
    h, w, c = imgs[0].shape
    images = torch.zeros((len(imgs), c, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.rollaxis(img, 2)
        images[i] += torch.from_numpy(nump_array)
    return images

def load_numpy_files(dirpath, load_fraction):
    images = []
    files = glob.glob(os.path.join(dirpath, "*.npy"))
    files.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
    if load_fraction:
        files = files[:1000]
    assert len(files) > 0, "Empty directory: " + dirpath
    for fpath in files:
        assert os.path.isfile(fpath), "Is not file: " + fpath
        ims = np.load(fpath)
        assert ims.dtype == np.uint8, "Was: {}".format(ims.dtype)
        images.append(ims)
    images = np.concatenate(images, axis=0)
    return images

def load_dataset_files(dirpath, imsize, load_fraction):
    images = load_numpy_files(os.path.join(dirpath, str(imsize)), load_fraction)
    return images

def _load_dataset(dirpath, imsize, batch_size, full_validation, load_fraction):
    images = load_dataset_files(dirpath, imsize, load_fraction)
    dataset_train = BaseDataset(images, augment_data=False)
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=16,
                                                   drop_last=True,
                                                   pin_memory=True,
                                                   collate_fn=fast_collate)
    dataloader_train = DataPrefetcher(dataloader_train)
    return dataloader_train


def load_mnist(batch_size, imsize=32):
    print("MNIST loaded, imsize:", imsize)
    transform = [
        transforms.Pad(2)
    ]
    if imsize != 32:
        transform +=  [transforms.Resize([imsize, imsize])]
    transform = transforms.Compose(transform)
    mnist_data = MNIST('data/mnist_data', 
                                train=True, 
                                download=True,
                                transform=transform)
    data_loader = torch.utils.data.DataLoader(mnist_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            drop_last=True,
                                            pin_memory=True,
                                            collate_fn=fast_collate)
    return data_loader


class MNIST(datasets.MNIST):
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target,mean_pixel) where target is index of the target class.
        """
        img, target = super(MNIST,self).__getitem__(index)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # the exta item to be returned
        return np.asarray(img)[:, :, None]

if __name__ == "__main__":
    import cv2
    import tqdm
    from src import torch_utils
    from src.data_tools.data_utils import denormalize_img
    from torch import nn
    
    
    imdir = "src/data_tools/.debug"
    os.makedirs(imdir, exist_ok=True)
    imsizes = [16, 32, 64]
    images = [[] for j in range(len(imsizes))]
    for im_idx, imsize in enumerate(imsizes):
        dl = load_dataset("celeba-HQ", 128, imsize, True, load_fraction=True)
        dl.update_next_transition_variable(1.0)
        ims = denormalize_img(next(iter(dl)))
        images[im_idx].extend([im for im in ims])
        ims = torch_utils.image_to_numpy(ims, to_uint8=True)
        for idx, im in enumerate(tqdm.tqdm(ims, desc="Saving images")):
            
            fpath = os.path.join(imdir, f'{idx}_{imsize}.jpg')
            cv2.imwrite(fpath, im[:, :, ::-1])
    for j in range(len(images[0])):
        im16 = images[0][j]
        im32 = images[1][j]
        im64 = images[2][j]
        im64 = nn.functional.avg_pool2d(im64, 2)
        print("Diff 32 vs 64:", abs(im64 - im32).sum())
        im32 = nn.functional.avg_pool2d(im32, 2)
        print("Diff 16 vs 32:", abs(im32 - im16).sum())



