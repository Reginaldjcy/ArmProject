# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from visualization_msgs.msg import Marker
# from scipy.spatial.transform import Rotation as R
# import numpy as np
# from .utils import create_arrow_marker  # 确保 utils 中有这个函数

# class MultiPosePublisher(Node):
#     def __init__(self):
#         super().__init__('multi_pose_publisher')
#         self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)
#         self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

#         # ✅ 定义三个目标点和方向
#         self.targets = [
#             (np.array([0.2, 0.3, 0.3]), np.array([1.0, 0.0, 0.0])),  # 指向 X 轴
#             (np.array([0.3, 0.0, 0.3]), np.array([0.0, 1.0, 0.0])),  # 指向 Y 轴
#             (np.array([0.3, -0.2, 0.3]), np.array([-1.0, 1.0, 0.0]))  # 指向 XY 斜方向
#         ]


#         self.current_index = 0
#         self.timer = self.create_timer(10.0, self.publish_next_pose)  # 每 10 秒发布一次

#         self.get_logger().info("🟢 开始循环发布三个目标 pose，每个间隔 10s")

#     def publish_next_pose(self):
#         if self.current_index >= len(self.targets):
#             self.get_logger().info("✅ 所有目标 pose 发布完成，准备关闭")
#             rclpy.shutdown()
#             return

#         start, direction = self.targets[self.current_index]
#         direction = direction / np.linalg.norm(direction)
#         z_axis = direction

#         # ✅ 发布箭头 Marker
#         marker = create_arrow_marker(start, direction)
#         self.marker_pub.publish(marker)

#         # ✅ 计算右手坐标系
#         tmp = np.array([0, 1, 0]) if abs(z_axis[1]) < 0.9 else np.array([1, 0, 0])
#         x_axis = np.cross(tmp, z_axis)
#         x_axis /= np.linalg.norm(x_axis)
#         y_axis = np.cross(z_axis, x_axis)
#         R_mat = np.column_stack((x_axis, y_axis, z_axis))

#         # ✅ 加入姿态补偿（绕 y 轴 +90°）
#         correction_rot = R.from_euler('y', np.pi / 2).as_matrix()
#         R_corrected = R_mat @ correction_rot
#         quat = R.from_matrix(R_corrected).as_quat()

#         # ✅ 构造 PoseStamped 并发布
#         msg = PoseStamped()
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.header.frame_id = 'base_link'
#         msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = start
#         msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat

#         self.publisher.publish(msg)
#         self.get_logger().info(f"📤 发布第 {self.current_index + 1} 个目标 pose ✅")
#         self.current_index += 1


# def main():
#     rclpy.init()
#     node = MultiPosePublisher()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()




import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np

class MultiPosePublisher(Node):
    def __init__(self):
        super().__init__('multi_pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)

        # ✅ 定义三个目标点（只包含位置，不包含方向）
        self.targets = [
            np.array([0.2, 0.3, 0.3]),
            np.array([0.3, 0.0, 0.3]),
            np.array([0.3, -0.2, 0.3])
        ]

        self.current_index = 0
        self.timer = self.create_timer(5.0, self.publish_next_pose)  # 每 10 秒发布一次
        self.get_logger().info("🟢 开始循环发布三个目标位置，每个间隔 10s")

    def publish_next_pose(self):
        if self.current_index >= len(self.targets):
            self.get_logger().info("✅ 所有目标位置发布完成，准备关闭")
            rclpy.shutdown()
            return

        position = self.targets[self.current_index]

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = position
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0  # 单位四元数，表示无旋转

        self.publisher.publish(msg)
        self.get_logger().info(f"📤 发布第 {self.current_index + 1} 个目标位置 ✅")
        self.current_index += 1


def main():
    rclpy.init()
    node = MultiPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
