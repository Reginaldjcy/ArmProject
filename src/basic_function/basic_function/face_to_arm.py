import rclpy
from rclpy.node import Node

from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Float32MultiArray

from scipy.spatial.transform import Rotation as R
import numpy as np
from .utils import *

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])
robot_position = np.array([0.5, 0.0, -0.5])
robot_radius = 0.5  # 机器人半径
face_dist = 0.5  # 人脸距离机器人0.5m


class Face2Arm(Node):
    def __init__(self):
        super().__init__('face_2_arm')

        # Subscription
        self.face_pose_sub = self.create_subscription(TimeFloat, 'hopenet_1', self.face_pose_callback, 10)
        self.face_point_sub = self.create_subscription(TimeFloat, 'pose_1', self.face_point_callback, 10)

        # publisher
        self.target_msg = self.create_publisher(TimeFloat, 'face2arm', 10)
        self.normal_pub = self.create_publisher(Marker, 'normal_marker', 10)
        self.target_vis = self.create_publisher(Marker, 'robot_target_vis', 10)
        self.robot_pub = self.create_publisher(Marker, 'robot_position', 10)


        # initial
        self.face_pose = None
        self.face_point = None  

    def face_pose_callback(self, msg):
        self.face_pose = np.array(msg.matrix.data).flatten()
        self.sub_result()
    def face_point_callback(self, msg):
        self.face_point = np.array(msg.matrix.data).reshape(-1, 3) 
        self.sub_result()

    def sub_result(self):
        if self.face_pose is None or self.face_point is None:
            return 
        # looking for face center point
        part_points = [0,1,2,3,4,5,6,7,8,9,10]
        face_point = self.face_point[part_points]
        face_point= Pixel2Optical(face_point , intrinsic)
        face_center = np.average(face_point, axis=0)

        # pose data
        x, y, z, roll, pitch, yaw = self.face_pose
        pseu_point = np.array([[x, y, z],
                               [0, 0, 0]])
        pseu_list = Pixel2Optical(pseu_point, intrinsic)
        x, y, z = pseu_list[0]
        rot = R.from_euler('yxz', [yaw+90, -pitch, roll],  degrees=True)  #roll, 90-pitch, 180-yaw
        rotation_matrix = rot.as_matrix()
        normal_vector = rotation_matrix[:, 2]
        norm_marker = create_arrow_marker([x, y, z], normal_vector)
        self.normal_pub.publish(norm_marker)  # blue

        # 计算沿着face_dirc方向延伸0.5m的点
        target_point = how2shootface(face_center, normal_vector, face_dist, robot_position, robot_radius)
        
        # 发布target_point
        target_msg = TimeFloat()
        target_msg.header = Header()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.matrix = Float32MultiArray()
        target_msg.matrix.data = target_point.flatten().tolist()
        self.target_msg.publish(target_msg)

        #  # 在Rviz中可视化target_point
        target_pub_points = Point(x=float(target_point[0]), y=float(target_point[1]), z=float(target_point[2]))
        marker = create_point_marker([target_pub_points], scale=0.08) # green
        self.target_vis.publish(marker)

        # show robot position
        rob_points = Point(x=float(robot_position[0]), y=float(robot_position[1]), z=float(robot_position[2]))
        rob_marker = create_point_marker([rob_points], scale=0.1, color=(1.0, 0.0, 0.0, 1.0))   # RED
        self.robot_pub.publish(rob_marker)


def main(args=None):
    rclpy.init(args=args)
    pose_keypoints_node = Face2Arm()
    rclpy.spin(pose_keypoints_node)
    pose_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()