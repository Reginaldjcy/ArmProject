import rclpy
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer

from msg_interfaces.msg import TimeFloat
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils import *
from builtin_interfaces.msg import Duration

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])

class spk_look_brd(Node):
    def __init__(self):
        super().__init__('spk_look_brd')

        # Subscriptions
        self.pose_sub = Subscriber(self, TimeFloat, 'hopenet_1')
        self.board_sub = Subscriber(self, TimeFloat, 'board_1')

        # Synchronize topics
        self.sync = ApproximateTimeSynchronizer(
            [self.pose_sub, self.board_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        # Publishers
        self.hopenet_pub = self.create_publisher(Marker, 'hopenet_arrow', 10)
        self.normal_pub = self.create_publisher(Marker, 'normal_arrow', 10)
        self.intersect_point_show = self.create_publisher(Marker, 'intersect_point', 10)
        self.intersect_result_pub = self.create_publisher(Bool, '/node_calculation/spk_look_brd', 10)

        # Initial parameters
        self.res = None
        self.intersect_point = None
        self.pose_point = None
        self.pose_normal = None
        self.board_point = None
        self.board_normal = None
        self.board_rw = None

    def sync_callback(self, pose_msg, board_msg):
        # === 读取 pose ===
        x, y, z, yaw, pitch, roll = pose_msg.matrix.data
        pseu_point = np.array([[x, y, z],
                               [0, 0, 0]])
        pseu_list = Pixel2World(pseu_point, intrinsic)
        x, y, z = pseu_list[0]
        self.pose_point = np.array([x, y, z])

        # 将欧拉角（角度制）转换为四元数 'xyz', [roll, pitch, yaw]
        rot = R.from_euler('xyz', [roll, -pitch, -yaw+180], degrees=True)
        quat = rot.as_quat()  # 返回 [x, y, z, w]
        rot = R.from_quat(quat)
        self.pose_normal = rot.apply([1, 0, 0])

        # publish arrow
        hopenet_arrow = create_hopenet_marker(x, y, z, quat)
        normal_marker = create_arrow_marker(self.pose_point, self.pose_normal) 
        self.hopenet_pub.publish(hopenet_arrow)
        self.normal_pub.publish(normal_marker)

        
        # === 读取 board 点云并拟合平面 ===
        board_1 = np.array(board_msg.matrix.data).reshape(-1, 3)
        self.board_rw = Pixel2World(board_1, intrinsic)

        board_instance = PlaneFitter(self.board_rw)
        self.board_point, self.board_normal = board_instance.fit_plane()


        # === 计算相交点 ===
        self.res, self.intersect_point = ray_intersects_plane(self.pose_point, self.pose_normal, self.board_rw)

        # 发布结果
        self.publisher_callback()

    def publisher_callback(self):
        if self.pose_point is None or self.pose_normal is None:
            return

        # 发布交点 Marker
        if self.intersect_point is not None:

            intersect_marker = create_point_marker(self.intersect_point)
            
            self.intersect_point_show.publish(intersect_marker)

        # 发布是否相交
        msg = Bool()
        msg.data = self.res
        self.intersect_result_pub.publish(msg)

        self.get_logger().info(f"Intersection result: {self.res}")

def main(args=None):
    rclpy.init(args=args)
    node = spk_look_brd()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
