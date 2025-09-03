import rclpy
from rclpy.node import Node

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from msg_interfaces.msg import TimeFloat
import numpy as np
from .utils import *

webcam_wh = np.array([1280, 720])

webcam_intrinsic = np.array([
    [954.46228283, 0.0, 310.94013938],
    [0.0, 956.68474763, 235.31295846],
    [0.0, 0.0, 1.0]
])

robot_position = np.array([0.5, 0.0, -0.5])
robot_radius = 0.5  # 机器人半径

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])



class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # Subscriptions
        self.create_subscription(TimeFloat, 'situation_choose', self.situation_callback, 10)
        self.create_subscription(TimeFloat, "pose_1", self.pose_callback, 10)
        self.create_subscription(TimeFloat, "jeff_1", self.board_callback, 10)
        self.target_sub = self.create_subscription(TimeFloat, 'robot_target_msg', self.rgb_callback, 10)

        # Publisher
        self.publisher_ = self.create_publisher(PoseStamped, '/piper_control/pose', 10)

        # Timer (optional, not used now)
        # self.timer = self.create_timer(1.0, self.timer_callback)

        # Initialize data containers
        self.situation = None
        self.pose = None
        self.board = None
        self.people = None

        # Camera calibration parameters
        self.cam_height = 720
        self.cam_width = 1280
        self.cam_depth = 5000

        # 在 __init__ 中加入：
        self.timer = self.create_timer(20.0, self.timer_callback)  

    def situation_callback(self, msg):
        self.situation = np.array(msg.matrix.data).reshape(1,2)

    def pose_callback(self, msg):
        self.pose = np.array(msg.matrix.data).reshape(-1, 3)

    def board_callback(self, msg):
        self.board = np.array(msg.matrix.data).reshape(-1, 3)

    # 添加新的定时器回调函数：
    def timer_callback(self):
        self.calc_callback()

    def rgb_callback(self, msg):
        self.people = np.array(msg.matrix.data).reshape(-1,3)

    # 修改 calc_callback 以防止打印过多
    def calc_callback(self):
        if self.situation is None or self.pose is None or self.board is None or self.people is None:
            #print(self.situation, self.pose, self.board)
            self.get_logger().info("Waiting for data...")
            return

        # #########################
        # self.situation = np.array([11.0, 0.0])
        # ##########################

        if np.array_equal(self.situation, np.array([[11.0, 0.0]])):
            target_point = np.mean(self.pose[:1], axis=0)
            label = "第一"

        elif np.array_equal(self.situation, np.array([[12.0, 0.0]])):
            world_point, pixel_point = spk_brd(self.pose, self.board)
            direction_2 = world_point - robot_position
            unit_dir_2 = direction_2 / np.linalg.norm(direction_2)
            intersection_2 = robot_position + robot_radius * unit_dir_2
            target_point = intersection_2
            label = "第二"           

        else:
            world_point, pixel_point = center(self.pose, self.board)
            direction_2 = world_point - robot_position
            unit_dir_2 = direction_2 / np.linalg.norm(direction_2)
            intersection_2 = robot_position + robot_radius * unit_dir_2
            target_point = intersection_2
            label = "第三种情况"

        # 发布目标点        
        diff = np.array([0.55, 0.067, 0.5])
        dot = self.people 
        robot_frame = pixel_to_robot_frame(dot, diff)
        self.get_logger().info(f"{label}: {robot_frame}")

        # ✅ 构建并发布 PoseStamped，仅包含位置，姿态为默认单位四元数
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.pose.position.x = robot_frame[0]
        msg.pose.position.y = robot_frame[1]
        msg.pose.position.z = robot_frame[2]

        # 姿态设为单位四元数（无旋转）
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        
        self.publisher_.publish(msg)

       

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
