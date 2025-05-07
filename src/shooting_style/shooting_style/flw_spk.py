import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from msg_interfaces.msg import TimeFloat
import numpy as np

class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # Subscriptions
        self.create_subscription(TimeFloat, 'situation_choose', self.situation_callback, 10)
        self.create_subscription(TimeFloat, "pose_1", self.pose_callback, 10)
        self.create_subscription(TimeFloat, "board_1", self.board_callback, 10)

        # Publisher
        self.publisher_ = self.create_publisher(PoseStamped, '/piper_control/pose', 10)

        # Timer (optional, not used now)
        # self.timer = self.create_timer(1.0, self.timer_callback)

        # Initialize data containers
        self.situation = None

        ##################
        self.situation = [11.0, 0.0]
        ###################


        self.pose = None
        self.board = None

        # Camera calibration parameters
        self.cam_height = 720
        self.cam_width = 1280
        self.cam_depth = 5000

        # 在 __init__ 中加入：
        self.timer = self.create_timer(1, self.timer_callback)  # 每0.5秒触发一次

    def situation_callback(self, msg):
        self.situation = msg.matrix.data

    def pose_callback(self, msg):
        self.pose = np.array(msg.matrix.data).reshape(-1, 3)

    def board_callback(self, msg):
        self.board = np.array(msg.matrix.data).reshape(-1, 3)

    # 添加新的定时器回调函数：
    def timer_callback(self):
        self.calc_callback()

    # 修改 calc_callback 以防止打印过多
    def calc_callback(self):
        if self.situation is None or self.pose is None or self.board is None:
            self.get_logger().info("Waiting for data...")
            return

        if self.situation == [11.0, 0.0]:
            target_point = np.mean(self.pose[:1], axis=0)
            label = "🧑"
        elif self.situation == [12.0, 0.0]:
            target_point = np.mean(self.pose[:14] + self.board[:14], axis=0)
            label = "🏫"
        else:
            target_point = np.array([0.15, 0.0, 0.2])
            label = "🔥"

        # Normalize
        target_point = np.array([
            (target_point[2] / self.cam_depth) * 1,
            ((target_point[0] - 0.5 * self.cam_width) / self.cam_width) * 1,
            ((target_point[1] - 0.5 * self.cam_height) / self.cam_height) * 0.5,
        ])

        self.get_logger().info(f"{label}: {target_point}")

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.pose.position = Point(x=target_point[0],
                                y=-target_point[1],
                                z=target_point[2])
        msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
