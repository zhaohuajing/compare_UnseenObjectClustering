#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

from sensor_msgs.msg import CameraInfo

def publish_rgbd():
    rospy.init_node('test_rgbd_publisher')
    
    rgb_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=1)
    depth_pub = rospy.Publisher('/camera/aligned_depth_to_color/image_raw', Image, queue_size=1)
    cam_info_pub = rospy.Publisher('/camera/color/camera_info', CameraInfo, queue_size=1)

    bridge = CvBridge()
    frame_id = "camera_color_optical_frame"

    # Load images
    img_path = '/home/csrobot/graspnet_ws/src/UnseenObjectClustering/data/demo'
    rgb_img = cv2.imread(f'{img_path}/000000-color.png')
    depth_img = cv2.imread(f'{img_path}/000000-depth.png', cv2.IMREAD_UNCHANGED)

    if rgb_img is None or depth_img is None:
        rospy.logerr("Image load failed.")
        return

    # Create dummy camera info
    cam_info_msg = CameraInfo()
    cam_info_msg.header.frame_id = frame_id
    cam_info_msg.width = rgb_img.shape[1]
    cam_info_msg.height = rgb_img.shape[0]
    cam_info_msg.K = [525.0, 0.0, 319.5,
                      0.0, 525.0, 239.5,
                      0.0, 0.0, 1.0]
    cam_info_msg.P = [525.0, 0.0, 319.5, 0.0,
                      0.0, 525.0, 239.5, 0.0,
                      0.0, 0.0, 1.0, 0.0]
    cam_info_msg.distortion_model = "plumb_bob"

    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        try: 
            timestamp = rospy.Time.now()

            rgb_msg = bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')
            rgb_msg.header.stamp = timestamp
            rgb_msg.header.frame_id = frame_id

            depth_msg = bridge.cv2_to_imgmsg(depth_img, encoding='16UC1')
            depth_msg.header.stamp = timestamp
            depth_msg.header.frame_id = frame_id

            cam_info_msg.header.stamp = timestamp

            rgb_pub.publish(rgb_msg)
            depth_pub.publish(depth_msg)
            cam_info_pub.publish(cam_info_msg)

            rospy.loginfo("Published RGB, depth, and camera_info")
            rate.sleep()
        except rospy.ROSInterruptException:
            break

if __name__ == '__main__':
    publish_rgbd()
