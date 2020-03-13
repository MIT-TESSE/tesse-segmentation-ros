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

""" Utilities for training a TESSE semantic segmentation model. """

import numpy as np
import torch
import torch.nn.functional as F

import segmentation_models_pytorch as smp

# classes used for the GOSEEK challenge
GOSEEK_CLASSES = [
    "floor",
    "ceiling",
    "wall",
    "monitor",
    "door",
    "table",
    "chair",
    "storage",
    "couch",
    "clutter",
    "treasure",
]


def calculate_iou(predictions, labels, n_classes):
    """ Calculates IoU of the class predictions against the
        ground truth for a given batch of segmentation predictions

    Parameters
    ----------
    predictions: torch.Tensor, shape= NxCxHxW
        One hot class predictions

    labels: torch.Tensor, shape=Nx1xHxW
        Truth labels

    n_classes: int, optional (default=2)
        Number of classes

    Returns
    -------
    numpy.ndarray, shape=('n_classes',)
        IoU for each class and the mean
    """
    if len(predictions.shape) == 4:
        _, class_predictions = predictions.max(1)
    else:
        class_predictions = predictions

    # intersection
    intersect = class_predictions.clone()
    intersect[torch.ne(class_predictions.long(), labels.long())] = -1

    area_intersect = torch.histc(
        intersect.float(), min=0, bins=n_classes, max=n_classes - 1
    )

    # union
    class_predictions[torch.lt(labels, 0)] = -1
    area_predictions = torch.histc(
        class_predictions.float(), min=0, bins=n_classes, max=n_classes - 1
    )
    area_labels = torch.histc(labels.float(), min=0, bins=n_classes, max=n_classes - 1)
    area_overlap = area_predictions + area_labels - area_intersect

    return np.array(area_intersect / (area_overlap + 1e-10))


def cross_entropy(input, target, weights=None):
    """ Cross entropy loss over semantic segmentation data.

    Args:
        input (np.ndarray): Shape (H, W, C) input prediction.
        target (np.ndarray): Shape (H, W, C) one hot target vector.
        weights (Optional[np.ndarray]): Shape (C,) loss rescaling weights.

    Returns:
        torch.Tensor: Cross entropy loss.
    """
    b, c, h, w = input.shape
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    return F.cross_entropy(input, target, weight=weights)


class CrossEntropy(smp.utils.base.Loss):
    """ Wrapper for cross entropy loss. """

    def __init__(self, weights=None):
        """ Cross entropy loss.

        Args:
            weights (Optional[torch.Tensor]): Cross entropy class weighting.
        """
        super().__init__()
        self.weights = weights

    def forward(self, y_pr, y_gt):
        y_gt = y_gt.argmax(axis=1).long()
        return cross_entropy(y_pr, y_gt, weights=self.weights)
