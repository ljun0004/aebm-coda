import os
import numpy as np

import torch
import torchvision.datasets as datasets


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
            mode: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

        self.mode = mode

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)

        # print(f"CachedFolder - mode: {self.mode}")

        if self.mode == "kl":
            if torch.rand(1) < 0.5:  # randomly hflip
                moments = data['moments']
            else:
                moments = data['moments_flip']

            return moments, target
        
        elif self.mode == "coda":
            if torch.rand(1) < 0.5:  # randomly hflip
                moments = data['moments']
                gt_indices = data['gt_indices']
            else:
                moments = data['moments_flip']
                gt_indices = data['gt_indices_flip']

            return (moments, gt_indices), target

        elif self.mode == "vq":
            if torch.rand(1) < 0.5:  # randomly hflip
                h = torch.from_numpy(data['h'])
                gt_indices = torch.from_numpy(data['gt_indices'])
            else:
                h = torch.from_numpy(data['h_flip'])
                gt_indices = torch.from_numpy(data['gt_indices_flip'])

            return (h, gt_indices), target

        else:
            raise NotImplementedError