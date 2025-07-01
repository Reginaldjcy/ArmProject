import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

from cv_bridge import CvBridge
import cv2
import numpy as np
from .utils import *

class face2arm(Node):
    def __init__(self):
        super().__init__("cam2arm")  # node name

        # Subscription
        self.target_sub = self.create_subscription(Float32MultiArray, 'pose_1', self.face_callback, 10)

        # Publisher
        self.arm_pub = self.create_publisher(Float32MultiArray, 'face2arm', 10)
    
    def face_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        human_1 = np.array(msg.matrix.data).reshape(-1,3)
        

        # Convert dot to realworld coordinates
        dot_wrd = Pixel2World(dot, intrinsic)
        print("dot_wrd:", dot_wrd)  

        # From pixel to robot frame
        robot_frame = pixel_to_robot_frame(dot_wrd, diff)
        print("robot_frame:", robot_frame)



def main():
    rclpy.init()
    node = face2arm()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
