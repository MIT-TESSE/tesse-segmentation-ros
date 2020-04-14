#! /usr/bin/env python

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

from os import path


import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge

from tesse_segmentation_ros.models import get_model
from tesse_segmentation_ros.utils import get_class_colored_image


class SemanticSegmentation:
    n_channels = 3

    def __init__(self):
        """ Semantic segmentation node. """

        # get model type
        model = rospy.get_param("~model", "TRTModel")

        # get model weight file
        model_weight_path = rospy.get_param("~weight_file", "")

        # true to publish an RGB image with color corresponding to class values
        # otherwise, just publish 1 channel image with integer class values.
        self.publish_color_image = rospy.get_param("~class_colored_image", True)

        # check for empty or non-existing parameters
        if model is "" or model_weight_path is "":
            raise ValueError("Must provide a value for param `~model` and `~model_path`")
        if not path.exists(model_weight_path):
            raise ValueError("Model weight path: %s does not exist" % model_weight_path)

        self.model = get_model(model, model_weight_path)
        rospy.loginfo("model initialized")

        self.image_subscriber = rospy.Subscriber(
            "/image/image_raw", Image, self.image_callback
        )
        self.image_publisher = rospy.Publisher("~prediction", Image, queue_size=10)

        if self.publish_color_image:
            self.class_color_image_publisher = rospy.Publisher(
                "~prediction_class_colored", Image, queue_size=10
            )

        self.cv_bridge = CvBridge()
        self.last_image_and_timestamp = None
        self.predict = False

        self.spin()

    def image_callback(self, image_msg):
        """ Record latest image and timestamp. Latest image will be
        processed in main thread, `spin`. """
        self.last_image_and_timestamp = self.decode_image(image_msg)
        self.predict = True

    def spin(self):
        """ Perform semantic segmentation and publish image. """
        while not rospy.is_shutdown():
            if self.predict:
                img, timestamp = self.last_image_and_timestamp
                prediction = self.model.infer(img)
                prediction = prediction.astype(np.uint8)

                self.image_publisher.publish(
                    self.get_image_message(prediction, timestamp)
                )

                if self.publish_color_image:
                    self.class_color_image_publisher.publish(
                        self.get_image_message(
                            get_class_colored_image(prediction), timestamp, "rgb8"
                        )
                    )

                self.predict = False

    def decode_image(self, image_msg):
        """ Decode image message to image and timestamp.

        Args:
            image_msg (Image): ROS Image message.

        Returns:
            Tuple[np.ndarray, time]: Image array and ROS time message.
        """
        height, width = image_msg.height, image_msg.width
        img = (
            np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                (height, width, self.n_channels)
            )
            / 255.0
        )
        return img, image_msg.header.stamp

    def get_image_message(self, image, timestamp, encoding="mono8"):
        """ Convert image array and timestamp to ROS image message.

        Args:
            image (np.ndarray): Shape (H, W, C) or (H, W) image array.
            timestamp (time): ROS timestamp.
            encoding (Optional[str]): Optional encoding. Default is "mono8".

        Returns:
            Image: ROS image message.
        """
        img_msg = self.cv_bridge.cv2_to_imgmsg(image.astype(np.uint8), encoding)
        img_msg.header.stamp = timestamp
        return img_msg


if __name__ == "__main__":
    rospy.init_node("SemanticSegmentation_node")
    node = SemanticSegmentation()
    rospy.spin()
