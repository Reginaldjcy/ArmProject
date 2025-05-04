import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import cv2

class RGBDMeasurement(Node):
    def __init__(self):
        super().__init__('rgbd_measurement')

        # 创建 RGB 和深度图像订阅者
        self.rgb_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, '/camera/depth/image_raw', self.depth_callback, 10
        )

        # 创建 ROS 到 OpenCV 的桥接器
        self.bridge = CvBridge()

        # 用于存储 RGB 和深度图像
        self.rgb_image = None
        self.depth_image = None

    def rgb_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 图像
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.get_logger().error(f"RGB convert")
        except Exception as e:
            self.get_logger().error(f"Could not convert RGB image: {e}")

    def depth_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 OpenCV 深度图像
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            self.get_logger().error(f"depth convert")
        except Exception as e:
            self.get_logger().error(f"Could not convert Depth image: {e}")

        # 确保 RGB 和深度图像都已接收
        if self.rgb_image is not None and self.depth_image is not None:
            self.measure_bounding_box()

    def measure_bounding_box(self):
        # 将深度图像转换为点云
        intrinsic_matrix = np.array([
            [570, 0, 320],
            [0, 570, 240],
            [0, 0, 1]
        ])  # 示例相机内参，根据你的相机调整

        # 深度图像的高度和宽度
        height, width = self.depth_image.shape

        # 转换为点云
        points = []
        for v in range(height):
            for u in range(width):
                z = self.depth_image[v, u]
                if z > 0:  # 忽略没有深度值的点
                    x = (u - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
                    y = (v - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
                    points.append([x, y, z])

        # 使用 Open3D 构建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 估计包裹物的边界框
        axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
        bounding_box = axis_aligned_bounding_box.get_box_points()

        # 计算包裹的长宽高
        min_bound = axis_aligned_bounding_box.min_bound
        max_bound = axis_aligned_bounding_box.max_bound
        length = max_bound[0] - min_bound[0]
        width = max_bound[1] - min_bound[1]
        height = max_bound[2] - min_bound[2]

        self.get_logger().info(f"Bounding Box Length: {length:.2f} meters")
        self.get_logger().info(f"Bounding Box Width: {width:.2f} meters")
        self.get_logger().info(f"Bounding Box Height: {height:.2f} meters")

        # 可视化包裹的点云和边界框
        o3d.visualization.draw_geometries([pcd, axis_aligned_bounding_box])

def main(args=None):
    rclpy.init(args=args)
    node = RGBDMeasurement()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
