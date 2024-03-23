#!/usr/bin/env python3
import rodent
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import rospy
import numpy as np
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3, TwistWithCovarianceStamped
import threading

depth = None
flow = Vector3()
twist = TwistWithCovarianceStamped()

def depth_data_callback(msg):
    global depth
    depth = msg.data
    # rospy.loginfo("depth: %f", depth)


tracker = rodent.RodentTracker(resize_height=300, visualize=True, feature_history=5, odometry_history=20)
bridge = CvBridge()

# Coeffecients and matrix from the camera when outside the hull
camera_matrix = np.array([[1706.8565539611177, 0.0, 580.1391682370544], [0.0, 1550.5616213728792, 319.5774072691713], [0.0, 0.0, 1.0]])
dist_coeffs = np.array([0.30478064767638086, -14.56299817406861, 0.015284616557577364, -0.04791390444929139, 76.17203511800008])

flow_publisher = rospy.Publisher("/rodent/flow", Vector3, queue_size=10)
twist_publisher = rospy.Publisher("/rodent/twist", TwistWithCovarianceStamped, queue_size=10)


def publish_twist():
    global twist

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        delta = now - twist.header.stamp
        twist.header.stamp = now

        div = 10 ** 7 / delta.to_nsec()
        twist.twist.twist.linear.x *= div
        twist.twist.twist.linear.y *= div

        twist_publisher.publish(twist)
        twist.twist.twist.linear.x = 0
        twist.twist.twist.linear.y = 0

        rate.sleep()


def on_image(img):
    global flow, twist

    if not depth:
        return
    
    frame = bridge.imgmsg_to_cv2(img)
    # frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tracker.read_frame(frame)
    raw_flow = tracker.run_frame()
    
    # Multiply flow elements by 0.296 * depth
    scaled_flow = raw_flow * (0.296 * (3 - 0.6 - depth))
    
    flow.x = scaled_flow[0]
    flow.y = scaled_flow[1]

    flow_publisher.publish(flow)

    twist.twist.twist.linear.x += scaled_flow[0]
    twist.twist.twist.linear.y += scaled_flow[1]


depth_sub = rospy.Subscriber("/depth_data", Float64, depth_data_callback)
image_sub = rospy.Subscriber('/emulation/camera/image_color', Image, on_image)




if __name__ == "__main__":
    rospy.init_node("rodent")

    twist_thread = threading.Thread(target=publish_twist, daemon=True)
    twist_thread.start()

    rospy.spin()

