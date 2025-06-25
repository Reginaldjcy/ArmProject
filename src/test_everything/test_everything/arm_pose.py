import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation as R
import numpy as np
from .utils import *

class PiperPosePublisher(Node):
    def __init__(self):
        super().__init__('piper_pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        # ✅ 设置目标位置和方向
        start = np.array([0.1, 0.3, 0.3])                     # 目标位置
        direction = np.array([0.5, -0.5, 0.2])                # 指定方向
        direction = direction / np.linalg.norm(direction)    # 单位化
        z_axis = direction                                   # 末端 z 轴朝向 -direction

        marker = create_arrow_marker(start, direction)
        self.marker_pub.publish(marker)

        # ✅ 构造正交右手坐标系
        tmp = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
        x_axis = np.cross(tmp, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R_mat = np.column_stack((x_axis, y_axis, z_axis))

        # ✅ 加入姿态补偿（因为 Arm_IK 中 ee frame 是绕 X 轴 -90°）
        correction_rot = R.from_euler('y', +np.pi / 2).as_matrix()    #R.from_euler('y', +np.pi / 2)
        R_corrected = R_mat @ correction_rot

        quat = R.from_matrix(R_corrected).as_quat()

        # ✅ 构建并发布 PoseStamped
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.position.x = start[0]
        msg.pose.position.y = start[1]
        msg.pose.position.z = start[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        self.get_logger().info("📤 发布修正后的目标 pose ✅（考虑了 ee 补偿）")
        self.publisher.publish(msg)

        # 稍作延时后关闭
        self.create_timer(1.0, rclpy.shutdown)


def main():
    rclpy.init()
    node = PiperPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
