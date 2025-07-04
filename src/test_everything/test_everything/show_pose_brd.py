import rclpy
from rclpy.node import Node

from msg_interfaces.msg import TimeFloat
from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np
from .utils import *

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


class TestGetPlane(Node):
    def __init__(self):
        super().__init__('publish_pose_brd')

        # Subscription
        self.pose_sub = Subscriber(self, TimeFloat, 'pose_1')
        self.brd_sub = Subscriber(self, TimeFloat, 'board_1')

        # Synchronize two topic
        self.sync = ApproximateTimeSynchronizer(
            [self.pose_sub, self.brd_sub],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.sync_callback)

        # 发布 arrow
        self.pose_arrow = self.create_publisher(Marker, 'pose_arrow', 10)
        self.board_plane = self.create_publisher(Marker, 'brd_plane', 10)
  
    def sync_callback(self, pose_msg, brd_msg):
        pose_pixel = np.array(pose_msg.matrix.data).reshape(-1,3)
        brd_pixel = np.array(brd_msg.matrix.data).reshape(-1,3)

        ## chose human body 
        pose_pixel = pose_pixel[[11,12,24,23]]

        # 拟合平面
        pose_wrd= Pixel2Rviz(pose_pixel , intrinsic)
        brd_wrd = Pixel2Rviz(brd_pixel , intrinsic)
        plane_fitter_1 = PlaneFitter(pose_wrd)
        point_pose, normal_pose = plane_fitter_1.fit_plane()
        plane_fitter_2 = PlaneFitter(brd_wrd)
        point_brd, normal_brd = plane_fitter_2.fit_plane()

        ########### make normal vector always front ##########
        if normal_pose[0] > 0:
            normal_pose = -normal_pose
        if normal_brd[0] > 0:
            normal_brd = -normal_brd

        ########## for arrow and plane publish #########
        if normal_pose is None or normal_brd is None:
            return
        
        # 计算法向量与坐标轴夹角（单位：度）
        def compute_angles(normal):
            norm = np.linalg.norm(normal)
            angle_x = np.arccos(normal[0] / norm) * 180 / np.pi
            angle_y = np.arccos(normal[1] / norm) * 180 / np.pi
            angle_z = np.arccos(normal[2] / norm) * 180 / np.pi
            return angle_x, angle_y, angle_z

        angles_pose = compute_angles(normal_pose)


        # 打印结果
        self.get_logger().info(f'[Pose Plane] 夹角 x: {angles_pose[0]:.2f}°, y: {angles_pose[1]:.2f}°, z: {angles_pose[2]:.2f}°')

        ### publish arrow
        pose_arrow = create_arrow_marker(point_pose, normal_pose)

        self.pose_arrow.publish(pose_arrow)

        ### publish plan
        brd_plane = create_plane_marker(brd_wrd)

        self.board_plane.publish(brd_plane)



def main():
    rclpy.init()
    node = TestGetPlane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
