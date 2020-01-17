###################################################################################################
# Distribution authorized to U.S. Government agencies and their contractors. Other requests for
# this document shall be referred to the MIT Lincoln Laboratory Technology Office.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#

# (c) 2019 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

import numpy as np


# generic labels from the cityscapes dataset
SEGMENTATION_COLORS = np.array(
    [
        [128, 64, 128],
        [244, 35, 2320],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
)


# colors used in TESSE v53
SEGMENTATION_COLORS_V53 = np.array(
    [
        [0, 171, 143],
        [1, 155, 211],
        [2, 222, 110],
        [3, 69, 227],
        [4, 218, 221],
        [5, 81, 38],
        [6, 229, 176],
        [7, 106, 60],
        [8, 179, 10],
        [9, 118, 90],
        [10, 138, 80],
    ]
)


def get_debug_image(image):
    """ Turn color code labels.

    Args:
        image (np.ndarray): HxW label image.

    Returns:
        np.ndarray: Color coded RGB image.
    """
    labels = np.unique(image)
    if labels.shape[0] > SEGMENTATION_COLORS.shape[0]:
        raise ValueError("Need more segmentation colors")

    color_image = np.zeros(image.shape + (3,))
    for i, label in enumerate(labels):
        color_image[np.where(image == label)] = SEGMENTATION_COLORS_V53[label]
    return color_image


def pad_image(img, h_pad, w_pad):
    """Add a padding of (`h_pad`, `w_pad`) to `img`

    Args:
        img (np.ndarray): Array of shape (h, w, c)
        h_pad (int): Total width padding
        w_pad (int): Total width padding

    Returns:
        np.ndarray: Array of shape (h+h_pad, w+w_pad, c)
    """
    assert h_pad % 2 == 0 and w_pad % 2 == 0
    padding = ((h_pad // 2, h_pad // 2), (w_pad // 2, w_pad // 2))

    if len(img.shape) == 3:
        padding += ((0, 0),)

    return np.pad(img, padding, mode="constant")


def unpad_image(img, h_pad, w_pad):
    """Remove the edge (`h_pad`, `w_pad`) indicies of `img`

    Args:
        img (np.ndarray): Array of shape (h+h_pad, w+w_pad, c)
        h_pad (int): Height padding to remove.
        w_pad (int): Width padding to remove.

    Returns:
        np.ndarray: Array of shape (h, w, c)
    """
    assert h_pad % 2 == 0 and w_pad % 2 == 0
    h, w = img.shape[:2]
    return img[h_pad // 2 : h - (h_pad // 2), w_pad // 2 : w - (w_pad // 2)]
