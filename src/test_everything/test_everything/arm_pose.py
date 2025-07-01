import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class PiperPosePublisher(Node):
    def __init__(self):
        super().__init__('piper_pose_publisher')
        self.publisher = self.create_publisher(PoseStamped, 'piper_control/pose', 10)

        # ✅ 设置目标位置（只包含位置坐标）
        self.target_position = [0.416, 0.212, 0.284]#####[0.483, -0.343, 0.426]  # 你的末端示教点位置

        # ✅ 定时持续发布，每0.1秒一次
        self.timer_period = 1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.publish_pose)

    def publish_pose(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.position.x = self.target_position[0]
        msg.pose.position.y = self.target_position[1]
        msg.pose.position.z = self.target_position[2]

        # 姿态设为单位四元数（无旋转）
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)
        self.get_logger().info("📤 持续发布目标 pose ✅")


def main():
    rclpy.init()
    node = PiperPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
