import rclpy
from rclpy.node import Node

from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np
from .utils import *

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])

class TestGetPoint(Node):
    def __init__(self):
        super().__init__('publish_show_point')

        # subscription
        self.subscription = self.create_subscription(TimeFloat, 'board_1', self.subscription_callback, 10)

        # publisher
        self.publisher_ = self.create_publisher(Marker, 'preset_brd', 10)

        self.points = []

    def subscription_callback(self, msg):
        # Get data
        pose_1 = np.array(msg.matrix.data).reshape(-1,3)
        pose_1 = Pixel2Rviz(pose_1, intrinsic)


        self.points = pose_1
        self.get_logger().info(f"publish {self.points}")
        self.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.points]

        # Publish the keypoints data
        self.publisher_marker()

    def publisher_marker(self):
        if self.points is not None:
            marker = create_plane_marker(self.points)
            self.publisher_.publish(marker)

              
def main(args=None):
    rclpy.init(args=args)
    pose_keypoints_node = TestGetPoint()
    rclpy.spin(pose_keypoints_node)
    pose_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()