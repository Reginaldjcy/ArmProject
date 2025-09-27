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
        self.subscription = self.create_subscription(TimeFloat, 'pose_1', self.subscription_callback, 10)

        # publisher
        self.publisher_ = self.create_publisher(Marker, 'point_1', 10)

        self.points = []
        self.keypoint_part = 'face'

    def subscription_callback(self, msg):
        # Get data
        pose_1 = np.array(msg.matrix.data).reshape(-1,3)
        pose_1 = Pixel2Optical(pose_1, intrinsic)

        # select points
        if self.keypoint_part == 'face': 
            part_points = [0,1,2,3,4,5,6,7,8,9,10]
        elif self.keypoint_part == 'body':
            part_points = [11,12,24,23]
        elif self.keypoint_part == 'hand':
            part_points = None

        self.points = pose_1#[part_points]

        # self.points = np.array([[0, 0, 1],
        #                         [0, 0, 0]])

        # self.get_logger().info(f"publish {self.points}")
        self.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.points]

        # Publish the keypoints data
        self.publisher_marker()

    def publisher_marker(self):
        if self.points is not None:
            marker = create_point_marker(self.points, color=(1.0, 0.0, 0.0, 1.0), scale=0.01)
            self.publisher_.publish(marker)

              
def main(args=None):
    rclpy.init(args=args)
    pose_keypoints_node = TestGetPoint()
    rclpy.spin(pose_keypoints_node)
    pose_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()