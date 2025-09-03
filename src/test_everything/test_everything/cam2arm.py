import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from msg_interfaces.msg import TimeFloat

from cv_bridge import CvBridge
import cv2
import numpy as np
from .utils import *

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])

class Cam2Arm(Node):
    def __init__(self):
        super().__init__("cam2arm")  # node name

        # Subscription
        self.target_sub = self.create_subscription(TimeFloat, 'robot_target_msg', self.rgb_callback, 10)

        # Publisher
        self.arm_pub = self.create_publisher(PoseStamped, 'piper_control/pose', 10)
    
    def rgb_callback(self, msg):
        # Preset dot and diff
        diff = np.array([0.55, 0.067, 0.5])

        dot = np.array(msg.matrix.data).reshape(-1,3)

        # From pixel to robot frame
        robot_frame = pixel_to_robot_frame(dot, diff)

        # ✅ 构建并发布 PoseStamped，仅包含位置，姿态为默认单位四元数
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.position.x = robot_frame[0]
        msg.pose.position.y = robot_frame[1]
        msg.pose.position.z = robot_frame[2]

        # 姿态设为单位四元数（无旋转）
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.arm_pub.publish(msg)

def main():
    rclpy.init()
    node = Cam2Arm()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
