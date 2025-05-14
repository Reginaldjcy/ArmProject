import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from msg_interfaces.msg import TimeFloat
from functools import partial

from cv_bridge import CvBridge
import numpy as np
from .utils import *
import cv2

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])



class PosePublisher(Node):
    def __init__(self):
        super().__init__('pose_publisher')

        # Subscriptions
        self.create_subscription(TimeFloat, 'situation_choose', partial(self.generic_callback, data_type="situation"), 10)
        self.create_subscription(TimeFloat, "pose_1", partial(self.generic_callback, data_type="pose"), 10)
        self.create_subscription(TimeFloat, "jeff_1", partial(self.generic_callback, data_type="board"), 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.camera_callback, 10)
        # Publisher
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)

        # Initialize data containers
        self.situation = None
        self.situation = np.array([11.0, 0.0])
        self.pose = None
        self.board = None
        self.test_point = np.array([[29, 187, 1991],
                                    [0, 0, 0]])
        self.color_image = None

        # time publish
        self.timer = self.create_timer(2, self.calc_callback)  

        # CV2 bridge
        self.bridge = CvBridge()

    def camera_callback(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def generic_callback(self, msg, data_type):
        array = np.array(msg.matrix.data).reshape(-1, 3) if data_type != "situation" else np.array(msg.matrix.data).reshape(1, 2)

        if data_type == "situation":
            self.situation = array
        elif data_type == "pose":
            self.pose = array
        elif data_type == "board":
            self.board = array


    # calculate robot end effect position and publish 
    def calc_callback(self):
        if self.situation is None or self.pose is None or self.board is None:
            self.get_logger().info("Waiting for data...")
            return
        
        # Choose situation and get world target point
        if np.allclose(self.situation, np.array([[11.0, 0.0]]), atol=1e-4):
            world_point, pixel_point = flw_spk(self.pose, self.board)
        elif np.allclose(self.situation, np.array([[12.0, 0.0]]), atol=1e-4):
            world_point, pixel_point = spk_brd(self.pose, self.board)
        else:
            world_point, pixel_point = center(self.test_point)

        # World point to robot workspace
        robot_cmd = World2Robot(world_point)
        joint_1, joint_2, joint_3, joint_5 = robot_cmd

        cv2.circle(self.color_image, tuple(np.round(pixel_point[0][:2]).astype(int)), radius=3, color=(0, 255, 0), thickness=-1)  # 绿色实心圆点
        

        # publish joint state
        msg = JointState()
        msg.header = Header()
        now = self.get_clock().now().to_msg()
        msg.header.stamp = now
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']   # joint_1 = +-0.7, joint5 = 0.349-0.125
        msg.position = [float(joint_1), float(joint_2), float(joint_3), 0.0, float(joint_5), 0.17, 0.0]
        msg.velocity = []
        msg.effort = []

        self.publisher_.publish(msg)
        self.get_logger().info(f"{msg.position}")

        cv2.imshow('Shooting point', self.color_image)
        cv2.waitKey(1)  

def main(args=None):
    rclpy.init(args=args)
    node = PosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()



