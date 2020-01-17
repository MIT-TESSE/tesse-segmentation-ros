#! /usr/bin/env python

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

import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge

from semantic_segmentation_ros.models import get_model
from semantic_segmentation_ros.utils import get_debug_image


class SemanticSegmentation:
    n_channels = 3

    def __init__(self):
        model = rospy.get_param("~model", "")
        model_weight_path = rospy.get_param("~weight_file", "")
        self.publish_debug_image = rospy.get_param("~debug_image", True)

        if model is "" or model_weight_path is "":
            raise ValueError("Must provide value for param `~model` and `~model_path`")

        self.model = get_model(model, model_weight_path)

        rospy.loginfo("model initialized")

        self.image_subscriber = rospy.Subscriber(
            "/image/image_raw", Image, self.image_callback
        )
        self.image_publisher = rospy.Publisher("/prediction", Image, queue_size=10)

        if self.publish_debug_image:
            self.debug_image_publisher = rospy.Publisher(
                "/prediction_debug", Image, queue_size=10
            )

        self.cv_bridge = CvBridge()
        self.last_image_timestamp = None
        self.predict = False

        self.spin()

    def image_callback(self, image_msg):
        self.last_image_timestamp = self.decode_image(image_msg)
        self.predict = True

    def spin(self):
        while not rospy.is_shutdown():
            if self.predict:
                img, timestamp = self.last_image_timestamp
                prediction = self.model.infer(img)
                prediction = prediction.astype(np.uint8)

                self.image_publisher.publish(
                    self.get_image_message(prediction, timestamp)
                )

                if self.publish_debug_image:
                    self.debug_image_publisher.publish(
                        self.get_image_message(
                            get_debug_image(prediction), timestamp, "rgb8"
                        )
                    )

                self.predict = False

    def decode_image(self, image_msg):
        height, width = image_msg.height, image_msg.width
        img = (
            np.frombuffer(image_msg.data, dtype=np.uint8).reshape(
                (height, width, self.n_channels)
            )
            / 255.0
        )
        return img, image_msg.header.stamp

    def get_image_message(self, image, timestamp, encoding="mono8"):
        img_msg = self.cv_bridge.cv2_to_imgmsg(image.astype(np.uint8), encoding)
        img_msg.header.stamp = timestamp
        return img_msg


if __name__ == "__main__":
    rospy.init_node("SemanticSegmentation_node")
    node = SemanticSegmentation()
    rospy.spin()
