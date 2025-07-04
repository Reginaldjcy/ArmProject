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


class FacePose(Node):
    def __init__(self):
        super().__init__('publish_face_pose')

        # Subscription
        self.face_pose_sub = self.create_subscription(TimeFloat, 'hopenet_1', self.face_pose_callback, 10)
        self.face_point_sub = self.create_subscription(TimeFloat, 'pose_1', self.face_point_callback, 10)

        # publisher
        self.publisher_ = self.create_publisher(TimeFloat, 'face_pose', 10)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

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
        face_point= Pixel2Rviz(face_point , intrinsic)
        face_center = np.average(face_point, axis=0)

        # pose data
        x, y, z, yaw, pitch, roll = self.face_pose
        rot = R.from_euler('zyx', [pitch, yaw, roll], degrees=True)
        correction = R.from_euler('x', -90, degrees=True)
        rot = rot * correction
        rotation_matrix = rot.as_matrix()
        normal_vector = rotation_matrix[:, 2]


        # 计算沿着face_dirc方向延伸0.5m的点
        target_point = face_center + 1 * normal_vector
        target_point = np.array((target_point, [0, 0, 0]))  # 添加齐次坐标

        # 发布target_point
        target_msg = TimeFloat()
        target_msg.header = Header()
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.header.frame_id = "base_frame"

        target_msg.matrix = Float32MultiArray()
        target_msg.matrix.data = target_point.flatten().tolist()

        self.publisher_.publish(target_msg)

        #  # 在Rviz中可视化target_point
        target_pub_points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in target_point]
        marker = create_point_marker(target_pub_points)
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    pose_keypoints_node = FacePose()
    rclpy.spin(pose_keypoints_node)
    pose_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()