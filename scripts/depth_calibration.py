#!/usr/bin/env python3
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2.aruco
from std_msgs.msg import Float64
from sshkeyboard import listen_keyboard, stop_listening
import pandas as pd

ARUDCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_TAG_WIDTH = 0.20000
CSV_OUTPUT = "depth_map.csv"

REGR_COEF = -0.00069833
REGR_INTERCEPT = 0.00201766
state = {"depth": None, "image": None}
records = []


def process_image(image):
    corners, ids, _ = cv2.aruco.detectMarkers(
        image, ARUDCO_DICT, parameters=ARUCO_PARAMS
    )

    aruco_tag_width = None

    if ids is not None:
        for i in range(len(ids)):
            marker_id = ids[i][0]
            corners_of_marker = corners[i][0]
            marker_size_pixels = max(corners_of_marker[:, 0]) - min(
                corners_of_marker[:, 0]
            )
            print(f"Marker {marker_id}: Size in pixels = {marker_size_pixels}")

            if aruco_tag_width is None or marker_size_pixels > aruco_tag_width:
                aruco_tag_width = marker_size_pixels

    return aruco_tag_width


def image_callback(msg):
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, "bgr8")
    state["image"] = image


def depth_data_callback(msg):
    state["depth"] = msg.data


def main():
    rospy.init_node("depth_calibration_node")
    rospy.Subscriber("/emulation/depth", Float64, depth_data_callback)
    rospy.Subscriber("/emulation/camera/image_color", Image, image_callback)

    rospy.spin()


def on_press(key):
    global records
    if key == "c":
        depth, image = state["depth"], state["image"]
        aruco_tag_width_in_px = process_image(image)

        if not aruco_tag_width_in_px:
            print("No Aruco tags detected in the image!")
            return

        px = ARUCO_TAG_WIDTH / aruco_tag_width_in_px
        records.append((depth, px))
        print("Captured!")

    if key == "s":
        data_frame = pd.DataFrame(records, columns=["depth", "px"])
        data_frame.to_csv(CSV_OUTPUT, index=False)

    if key == "q":
        stop_listening()


def main():
    rospy.init_node("depth_calibration_node")
    rospy.Subscriber("/emulation/depth", Float64, depth_data_callback)
    rospy.Subscriber("/emulation/camera/image_color", Image, image_callback)

    listen_keyboard(on_press=on_press)


if __name__ == "__main__":
    main()
