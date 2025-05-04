#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

import numpy as np
from .utils import *

class AABBMaskNode(Node):
    def __init__(self):
        super().__init__('aabb_mask_node')

        # Sub pointcloud
        self.obj_sub = self.create_subscription(Float32MultiArray, "/obj_point", self.obj_callback, 10)
        self.bot_sub = self.create_subscription(Float32MultiArray, "/bot_point", self.bot_callback, 10)

        # # pub aabb in rviz2
        #self.aabb_pub = self.create_publisher(Marker, "/aabb_marker", 10)

        # initial
        self.obj = None
        self.bot = None

    def obj_callback(self, msg):
        self.obj = np.array(msg.data).reshape(-1,3)
        if self.obj is not None:
            length, width = min_bounding_rectangle(self.obj[:, [1, 2]])
            #length, width = calculate_bounding_rectangle(self.obj[:, [1, 2]])

            print(f"估算的矩形长(m): {length}")
            print(f"估算的矩形宽(m): {width}")



    def bot_callback(self, msg):
        self.bot = np.array(msg.data).reshape(-1,3)

        if self.obj is not None and self.bot is not None:
            # 计算平面1的z值平均值
            z_avg_plane1 = np.mean(self.obj[:, 0])  # 取第3列（z值）

            # 计算平面2的z值平均值
            z_avg_plane2 = np.mean(self.bot[:, 0])  # 取第3列（z值）

            # 计算两个平面z值平均值之差
            z_difference = z_avg_plane1 - z_avg_plane2
            print(f"heigh is: {z_difference}")





def main(args=None):
    rclpy.init(args=args)
    node = AABBMaskNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
