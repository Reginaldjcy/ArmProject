import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from msg_interfaces.msg import TimeFloat
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

import numpy as np
from .utils import PlaneFitter, compute_plane_vertices, Pixel2World, ray_intersects_plane
import pandas as pd
import atexit

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])

class spk_look_brd(Node):
    def __init__(self):
        super().__init__('spk_look_brd')

        # Subscription
        self.human_sub = Subscriber(self, TimeFloat, 'pose_1')
        self.brd_sub = Subscriber(self, TimeFloat, 'board_1')

        # Synchronize two topic
        self.sync = ApproximateTimeSynchronizer(
            [self.human_sub, self.brd_sub],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.intersect_point_show = self.create_publisher(Marker, 'intersect_point', 10)
        self.intersect_result_pub = self.create_publisher(Bool, '/node_calculation/spk_look_brd', 10)

        # initial parameters
        self.res = None
        self.intersect_point = None
        self.pose_point = None
        self.pose_normal = None
        self.board_point = None
        self.board_normal = None
        self.board_rw = None
        self.board_start_point = None

    def sync_callback(self, pose_msg, board_msg):
        # Get data
        pose_1 = np.array(pose_msg.matrix.data).reshape(-1,3)
        board_1 = np.array(board_msg.matrix.data).reshape(-1,3)

        # select body 
        part_points = [11,12,24,23]
        pose_1 = pose_1[part_points]

        # Get Normal vector
        pose_rw = Pixel2World(pose_1, intrinsic)
        self.board_rw = Pixel2World(board_1, intrinsic)

        ### for pose
        pose_instance = PlaneFitter(pose_rw)
        self.pose_point, self.pose_normal = pose_instance.fit_plane()
        if self.pose_normal[0] > 0:
            self.pose_normal = -self.pose_normal
        
        ### for brd
        board_instance = PlaneFitter(self.board_rw)
        self.board_point, self.board_normal = board_instance.fit_plane()

        # determain intersect
        self.res, self.intersect_point = ray_intersects_plane(self.pose_point, self.pose_normal, self.board_rw)

        self.publisher_callback()
    
    def publisher_callback(self):
        if self.pose_point is None or self.pose_normal is None:
            return
        
        ##########################
        ##### intersect point ####
        if self.intersect_point is not None:
            marker = Marker()
            marker.header.frame_id = "camera_link"  # 设置参考坐标系
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "points"
            marker.id = 0
            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            # 添加一个点的位置
            point = Point()
            point.x = self.intersect_point[0]  # x坐标
            point.y = self.intersect_point[1]  # y坐标
            point.z = self.intersect_point[2]  # z坐标
            marker.points.append(point)
            
            # 设置点的大小
            marker.scale.x = 0.05  # 点的宽度（米）
            marker.scale.y = 0.05  # 点的高度（米）
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha值，1.0为完全不透明

            # 发布Marker
            self.intersect_point_show.publish(marker)

        ##########################
        #### intersect or not ####
        msg = Bool()
        msg.data = self.res
        self.intersect_result_pub.publish(msg)

        self.get_logger().info(f" Get result {self.res}")

def main(args=None):
    rclpy.init(args=args)
    my_node = spk_look_brd()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


