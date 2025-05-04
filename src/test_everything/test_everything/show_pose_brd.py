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
        pose_wrd= Pixel2World(pose_pixel , intrinsic)
        brd_wrd = Pixel2World(brd_pixel , intrinsic)
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
