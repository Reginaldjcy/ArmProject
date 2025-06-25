import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from msg_interfaces.msg import TimeFloat
from scipy.spatial.transform import Rotation as R
import numpy as np
from .utils import *

class Face2Arm(Node):
    def __init__(self):
        super().__init__('face2arm')
        # subscription
        self.face_pose_sub = self.create_subscription(TimeFloat, 'face_pose', self.sub_callback, 10)

        # publisher
        self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)

    def point_along_ray(origin, direction, distance):
        origin = np.array(origin)
        direction = np.array(direction)
        direction_unit = direction / np.linalg.norm(direction)  # 单位化方向向量
        target_point = origin + direction_unit * distance
        return target_point
    
    def sub_callback(self, msg):
        face_pose = np.array(msg.matrix.data).flatten()
        point = face_pose[:3]
        dirc = face_pose[-3:]
        dist = 0.5

        target_point = self.point_along_ray(point, dirc, dist)