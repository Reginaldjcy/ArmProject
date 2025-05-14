import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from .utils import *

class ClickPoint(Node):
    def __init__(self):
        super().__init__('face_keypoints')

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        # CV2 bridge
        self.bridge = CvBridge()

    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""
        # Convert ROS Image messages to OpenCV images
        self.img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        cv2.imshow("Image", self.img)
        cv2.setMouseCallback("Image", self.mouse_callback)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked pixel: (u={x}, v={y})")
            keypoints = np.array([[x, y],
                                 [0,0]])
            face_1 = get_depth(keypoints, self.depth_image, self.img)
            print(f"depth is {face_1}")
            # 后续可以发布成 ROS 消息

def main(args=None):
    rclpy.init(args=args)
    node = ClickPoint()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
