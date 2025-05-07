# piper_pose_publisher.py

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

class PiperPosePublisher(Node):
    def __init__(self):
        super().__init__('piper_pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        if self.count >= 10:
            rclpy.shutdown()
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'  # 请确保与你 URDF 根链接一致

        # ✅ 空间点：远离本体，略高
        msg.pose.position.x = 0.27
        msg.pose.position.y = 0.0
        msg.pose.position.z = 0.23

        # ✅ 姿态：手爪稍朝下但不翻转
        quat = R.from_euler('xyz', [0.0, 2.3, 0.0]).as_quat()
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        self.publisher.publish(msg)
        self.get_logger().info(f'Published target pose {self.count + 1}/10')
        self.count += 1


def main():
    rclpy.init()
    node = PiperPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
