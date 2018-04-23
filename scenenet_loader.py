# Code taken from https://github.com/jmccormac/pySceneNetRGBD/blob/master/calculate_optical_flow.py
# and from nyu_dataloader.py

import os
import os.path
import numpy as np
import torch.utils.data as data
from PIL import Image
import transforms


to_tensor = transforms.ToTensor()


def load_depth_map_in_m(file_name):
    image = Image.open(file_name)
    pixel = np.array(image)
    return pixel * 0.001


def load_trajectory_image(path, i):
    # subdirectories photo, depth, instance:
    image = Image.open(os.path.join(path, "photo", "%d.jpg" % (i*25)))
    rgb = np.array(image)
    depth = load_depth_map_in_m(os.path.join(path, "depth", "%d.png" % (i*25)))
    return rgb, depth


# Checks which trajectories there are
# {train,val}/{0, 1, ...,}/{0, 1, ...,}
def find_trajectories(dir):
    subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    trajectories = []
    for dir in subdirs:
        for d in sorted(os.listdir(dir)):
            trajectories.append(os.path.join(dir, d))
    trajectories.sort()
    return trajectories


# Walks a directory and finds all folder called depth
# Each trajectory has 300 images
# Load the images at given indices only
def make_dataset(dir, trajectory_to_idx):
    subdirs = [os.path.join(dir, d) for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    trajectories = []
    for dir in subdirs:
        # This level is named as the scene-id
        for d in sorted(os.listdir(dir)):
            path = os.path.join(dir, d)
            trajectories.append((path, trajectory_to_idx[d]))

    return trajectories


def train_transform(oheight, owidth):
    do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

    ts = [
        transforms.CenterCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)]

    return transforms.Compose(ts), transforms.Compose(ts + [transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)])


def val_transform(oheight, owidth):
    t = transforms.Compose([
        transforms.CenterCrop((oheight, owidth)),
    ])
    return t, t


class ScenenetDataset(data.Dataset):
    modality_names = ['rgb', 'rgbd', 'd'] # , 'g', 'gd'

    def __init__(self, root, type, trajectory_indices, oheight, owidth,
                 sparsifier=None, modality='rgb', loader=load_trajectory_image):
        self.trajectories = find_trajectories(root)
        print("Found %d trajectories" % len(self.trajectories))
        # Load all of the images in trajectory for now
        self.trajectory_indices = trajectory_indices
        self.oheight = oheight
        self.owidth = owidth
        self.transform = train_transform if type == 'train' else val_transform

        self.root = root

        if type not in ['train', 'val']:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = loader
        self.sparsifier = sparsifier

        if modality in self.modality_names:
            self.modality = modality
        else:
            raise (RuntimeError("Invalid modality type: " + modality + "\n"
                                "Supported dataset types are: " + ''.join(self.modality_names)))

    def create_sparse_depth(self, rgb, depth):
        if self.sparsifier is None:
            return depth
        else:
            mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
            sparse_depth = np.zeros(depth.shape)
            sparse_depth[mask_keep] = depth[mask_keep]
            return sparse_depth

    def create_rgbd(self, rgb, depth):
        sparse_depth = self.create_sparse_depth(rgb, depth)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        n_per_trajectory = len(self.trajectory_indices)
        path = self.trajectories[int(index / n_per_trajectory)]
        rgb, depth = self.loader(path, index % n_per_trajectory)
        return rgb, depth

    def __get_all_item__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor, input_np, depth_np)
        """
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            depth_transform, rgb_transform = self.transform(self.oheight, self.owidth)
            rgb_np = np.asfarray(rgb_transform(rgb), dtype=np.float) / 255.0
            depth_np = depth_transform(depth)
        else:
            raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np)
        elif self.modality == 'd':
            input_np = self.create_sparse_depth(rgb_np, depth_np)
        else:
            raise (RuntimeError("Invalid modality type: " + self.modality + "\n"
                                "Supported dataset types are: " + ''.join(self.modality_names)))

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor, input_np, depth_np

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (input_tensor, depth_tensor)
        """
        input_tensor, depth_tensor, input_np, depth_np = self.__get_all_item__(index)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.trajectories) * len(self.trajectory_indices)
