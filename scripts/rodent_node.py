#!/usr/bin/env python3
import os
import rodent
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import rospy
import numpy as np
import threading
from geometry_msgs.msg import Vector3, TwistWithCovarianceStamped


REGR_COEF = -0.00069833
REGR_INTERCEPT = 0.00201766
# PX = REGR_INTERCEPT + DEPTH


depth = None
flow = Vector3()
twist = TwistWithCovarianceStamped()
frame_count = 0


def publish_twist():
    global twist

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        delta = now - twist.header.stamp
        twist.header.stamp = now

        div = 10**7 / delta.to_nsec()
        twist.twist.twist.linear.x *= div
        twist.twist.twist.linear.y *= div

        twist_publisher.publish(twist)
        twist.twist.twist.linear.x = 0
        twist.twist.twist.linear.y = 0

        rate.sleep()


def depth_data_callback(msg):
    global depth
    depth = msg.data


def on_image(img):
    global flow, twist

    if not depth:
        return

    frame = bridge.imgmsg_to_cv2(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Save the frame
    save_frame(frame)

    tracker.read_frame(frame)
    raw_flow = tracker.run_frame()

    scaled_flow = raw_flow * (0.296 * (3 - 0.6 - depth))

    flow.x = scaled_flow[0]
    flow.y = scaled_flow[1]

    flow_publisher.publish(flow)

    twist.twist.twist.linear.x += scaled_flow[0]
    twist.twist.twist.linear.y += scaled_flow[1]


# function to save output
def save_frame(frame):
    global frame_count
    frame_count += 1
    filename = f"frame_{frame_count}.jpg"
    cv2.imwrite(os.path.join("frames", filename), frame)


if __name__ == "__main__":
    rospy.init_node("rodent")

    # Create the frames directory if it does not exist
    if not os.path.exists("frames"):
        os.makedirs("frames")

    bridge = CvBridge()
    tracker = rodent.RodentTracker(
        resize_height=300, visualize=True, feature_history=5, odometry_history=20
    )

    depth_sub = rospy.Subscriber("/emulation/depth", Float64, depth_data_callback)
    image_sub = rospy.Subscriber("/emulation/camera/image_color", Image, on_image)

    flow_publisher = rospy.Publisher("/rodent/flow", Vector3, queue_size=10)
    twist_publisher = rospy.Publisher(
        "/rodent/twist", TwistWithCovarianceStamped, queue_size=10
    )

    twist_thread = threading.Thread(target=publish_twist, daemon=True)
    twist_thread.start()

    rospy.spin()
