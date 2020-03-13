###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

""" Hold PyTorch dataset for TESSE semantic segmentation. """

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

from tesse.msgs import *
from tesse_gym.tasks.goseek import GoSeek


class RGBSegmentationEnv(GoSeek):
    """ Define a custom TESSE gym environment to provide RGB and segmentation. """

    N_CLASSES = 11
    WALL_CLS = 2

    @property
    def observation_space(self):
        """ This must be defined for custom observations. """
        return spaces.Box(-np.Inf, np.Inf, shape=(240, 320, 6))

    def form_agent_observation(self, tesse_data):
        """ Create the agent's observation from a TESSE data response.

        Args:
            tesse_data (DataResponse): TESSE DataResponse object containing
                RGB, depth, segmentation, and pose.

        Returns:
            np.ndarray: The agent's observation.
        """
        eo, seg = tesse_data.images
        observation = np.concatenate((eo, seg), axis=-1)
        return observation

    def observe(self):
        """ Get observation data from TESSE.

        Returns:
            DataResponse: TESSE DataResponse object.
        """
        cameras = [
            (Camera.RGB_LEFT, Compression.OFF, Channels.THREE),
            (Camera.SEGMENTATION, Compression.OFF, Channels.THREE),
        ]
        agent_data = self.env.request(DataRequest(metadata=True, cameras=cameras))
        return agent_data


class TESSEDataset(torch.utils.data.Dataset):
    BACKGROUND_CLASS = 255
    WALL_CLS = 2

    def __init__(self, imgs, labels, n_classes, preprocessor=None):
        """ Dataset for TESSE semantic segmentation

        Args:
            imgs (List[Union[str, Path]]):  List of images paths.
            labels (List[Union[str, Path]]): Corresponding label paths.
            n_classes (int): The number of classes.
            preprocessor (Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray])
                Optional method to preprocess image and label pair.

        """
        self.imgs = imgs
        self.labels = labels
        self.n_classes = n_classes
        self.preprocessor = preprocessor

        assert len(self.imgs) == len(self.labels)

    def __getitem__(self, idx):
        img = plt.imread(str(self.imgs[idx]))[..., :3]  # remove alpha channel
        label = plt.imread(str(self.labels[idx]))[..., 0]  # class is first channel

        # ensure label is uint8
        if not issubclass(label.dtype.type, np.integer):
            label = (255 * label).astype(np.uint8)

        if self.preprocessor:
            img, label = self.preprocessor(image=img, label=label)

        img = img.transpose(2, 0, 1).astype(np.float32)  # (h, w, c) -> (c, h ,w)

        # Unity labels non material pixels (e.g. space outside windows) as the
        # max image value.
        # Change this to the wall class
        label[label == self.BACKGROUND_CLASS] = self.WALL_CLS

        one_hot_label = (
            self.class_id_to_one_hot_vector(label).transpose(2, 0, 1).astype(np.float32)
        )
        return img, one_hot_label

    def class_id_to_one_hot_vector(self, label):
        """ Convert image of class IDs to one hot vectors.

        For each pixel value, `c`, convert to vector, `v`
        where:
            v[i] = 1 if i == c
            v[i] = 0 if i != c

        Args:
            label (np.ndarray): Shape (H, W) label image of class IDs.

        Returns:
            np.ndarray: Shape (H, W, C) image one hot vector image where `C`
                is the number of classes.
        """
        return np.arange(self.n_classes) == label[..., np.newaxis]

    def calculate_inverse_class_frequency(self):
        """ Calculate the inverse class frequency of this dataset.

        For each class, c in [0, C) where C is the total number of classes,
        compute the vector f defined as

            f[c] = 1 / p[c]

        where p[c] is the percent of the dataset that consists (pixel wise) of class c.

        Returns:
            torch.Tensor: Inverse class frequency.
        """
        class_counts = [0] * self.n_classes

        for label in self.labels:
            labels, counts = np.unique(
                (255 * plt.imread(str(label))[..., 0]).astype(np.uint8),
                return_counts=True,
            )

            for l, v in zip(labels, counts):
                l = 2 if l == 255 else l  # assign background to wall cls
                class_counts[l] += v

        inverse_class_frequecy = 1 / torch.Tensor(class_counts)
        inverse_class_frequecy /= inverse_class_frequecy.sum()
        return inverse_class_frequecy

    def __len__(self):
        return len(self.imgs)
