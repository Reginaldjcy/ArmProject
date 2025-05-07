#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_msgs.msg import Header

class PIPER(Node):
    def __init__(self):
        super().__init__('control_piper_node')

        self.pub_descartes = self.create_publisher(Pose, 'pos_cmd', 10)
        self.pub_joint = self.create_publisher(JointState, '/joint_ctrl_single', 10)    # '/joint_ctrl_single',
        self.left_pub_joint = self.create_publisher(JointState, '/left_joint_states', 100)
        self.right_pub_joint = self.create_publisher(JointState, '/right_joint_states', 100)

        self.descartes_msgs = Pose()
        timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)

    def init_pose(self):
        self.publish_joint(self.pub_joint, [0.0] * 7)
        self.get_logger().info("send joint control piper command")

    def left_init_pose(self):
        self.publish_joint(self.left_pub_joint, [0.0] * 7)
        self.get_logger().info("send joint control piper command")

    def right_init_pose(self):
        self.publish_joint(self.right_pub_joint, [0.0] * 7)
        self.get_logger().info("send joint control piper command")

    def joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        self.publish_joint(self.pub_joint, [j1, j2, j3, j4, j5, j6, gripper, -gripper])

    def left_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        self.publish_joint(self.left_pub_joint, [j1, j2, j3, j4, j5, j6, gripper])
        self.get_logger().info("send joint control piper command")

    def right_joint_control_piper(self, j1, j2, j3, j4, j5, j6, gripper):
        self.publish_joint(self.right_pub_joint, [j1, j2, j3, j4, j5, j6, gripper])
        self.get_logger().info("send joint control piper command")

    def publish_joint(self, publisher, positions):
        joint_states_msgs = JointState()
        joint_states_msgs.header = Header()
        joint_states_msgs.header.stamp = self.get_clock().now().to_msg()
        joint_states_msgs.name = [f'joint{i+1}' for i in range(len(positions))]
        joint_states_msgs.position = positions
        publisher.publish(joint_states_msgs)

    def descartes_control_piper(self, x, y, z, roll, pitch, yaw, gripper):
        # Placeholder for future Pose message construction
        # self.descartes_msgs.position.x = x
        # self.descartes_msgs.position.y = y
        # self.descartes_msgs.position.z = z
        # self.descartes_msgs.orientation = ...
        pass
    def timer_callback(self):
        # Example of how to use the functions
        # self.init_pose()
        # self.left_init_pose()
        # self.right_init_pose()
        self.joint_control_piper(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # self.left_joint_control_piper(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        # self.right_joint_control_piper(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5)
        pass


def main(args=None):
    rclpy.init(args=args)
    piper = PIPER()
    rclpy.spin(piper)
    piper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
