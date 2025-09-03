import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Point, Quaternion
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from scipy.spatial.transform import Rotation as R
import numpy as np
from .utils import Pixel2Rviz

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


class ArrowPublisher(Node):
    def __init__(self):
        super().__init__('arrow_publisher')

        # subscription
        self.subscription = self.create_subscription(TimeFloat, 'hopenet_1', self.subscription_callback, 10)

        # publisher
        self.marker_pub = self.create_publisher(Marker, 'hopenet_arrow', 10)
        self.quaternions_pub = self.create_publisher(TimeFloat, 'hopenet_quaternions', 10)


    def subscription_callback(self, msg):
        x, y, z, roll, pitch, yaw = msg.matrix.data    # rotation to rivz2
        pseu_point = np.array([[x, y, z],
                               [0, 0, 0]])
        pseu_list = Pixel2Rviz(pseu_point, intrinsic)
        x, y, z = pseu_list[0]

        # 将欧拉角（角度制）转换为四元数 'xyz', [roll, pitch, yaw]
        rot = R.from_euler('xyz', [roll, -pitch, -yaw+180], degrees=True)    #roll, -pitch, -yaw+180
        quat = rot.as_quat()  # 返回 [x, y, z, w]

        # 构建 Marker 消息
        marker = Marker()
        marker.header.frame_id = "camera_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "arrows"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # 设置位置
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # 设置朝向
        marker.pose.orientation = Quaternion(
            x=quat[0],
            y=quat[1],
            z=quat[2],
            w=quat[3]
        )

        # 设置箭头大小
        marker.scale.x = 0.5
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        # 设置颜色（红色箭头）
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # 发布 Marker
        self.marker_pub.publish(marker)
        self.get_logger().info("Arrow marker published.")

        # 假设 position 和 quat 是 numpy array 或 list
        position = [x, y, z]
        orientation = quat.tolist()  # [x, y, z, w]

        # 合并
        self.data = position + orientation

        # Publish the keypoints data
        self.publisher_callback()

    def publisher_callback(self):
        if self.data is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data
            self.data = np.array(self.data, dtype=np.float32)
            float_array = Float32MultiArray()
            float_array.data = self.data.flatten().tolist()
            msg.matrix = float_array
            
            self.quaternions_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArrowPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
