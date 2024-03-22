#!/usr/bin/env python3
import rodent
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import rospy
import numpy as np

tracker = rodent.RodentTracker(resize_height=480, visualize=True, feature_history=5, odometry_history=20)
bridge = CvBridge()


def on_image(img):
    frame = bridge.imgmsg_to_cv2(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tracker.read_frame(frame)
    tracker.run_frame()


image_sub = rospy.Subscriber('/emulation/camera/image_color', Image, on_image)


if __name__ == "__main__":
    rospy.init_node("rodent")
    rospy.spin()

