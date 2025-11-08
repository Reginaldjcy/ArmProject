#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2


class UndistortNode(Node):
    def __init__(self):
        super().__init__('undistort_node')
        self.bridge = CvBridge()

        # 输入输出话题
        self.sub_img = self.create_subscription(Image, '/camera/color/image_raw', self.img_cb, 10)
        self.sub_info = self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_cb, 10)
        self.pub = self.create_publisher(Image, '/camera/color/image_rect_user', 10)

        # 相机参数
        self.K, self.D, self.new_K = None, None, None

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)
        h, w = msg.height, msg.width
        self.new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0.5, (w, h))

    def img_cb(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        if self.K is not None and self.D is not None:
            rect = cv2.undistort(frame, self.K, self.D, None, self.new_K)
        else:
            rect = frame
        out = self.bridge.cv2_to_imgmsg(rect, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main():
    rclpy.init()
    node = UndistortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()






