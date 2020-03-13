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

import numpy as np


# generic label colors from the cityscapes dataset
SEGMENTATION_COLORS_CITYSCAPE = np.array(
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


# colors used in GOSEEK
SEGMENTATION_COLORS_GOSEEK = np.array(
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


def get_class_colored_image(image, color_map=SEGMENTATION_COLORS_GOSEEK):
    """ Turn color code labels.

    Args:
        image (np.ndarray): HxW label image.
        color_map (Optional[np.array]]) Shape (C, 3) array mapping
            each of C classes to a color.

    Returns:
        np.ndarray: Color coded RGB image.
    """
    labels = np.unique(image)
    if labels.shape[0] > color_map.shape[0]:
        raise ValueError(
            "Provided segmentation class color map does not have enough values"
        )

    color_image = np.zeros(image.shape + (3,))
    for i, label in enumerate(labels):
        color_image[np.where(image == label)] = color_map[label]
    return color_image
