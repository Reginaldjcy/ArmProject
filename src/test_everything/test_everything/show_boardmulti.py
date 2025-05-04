import rclpy
from rclpy.node import Node
from rclpy.timer import Timer

from msg_interfaces.msg import TimeFloat
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

import numpy as np
from .utils import PlaneFitter, compute_plane_vertices, Pixel2World, project_points_to_plane
import pandas as pd
import atexit

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


class TestGetPlane(Node):
    def __init__(self):
        super().__init__('publish_show_board')

        # subscription
        self.subscription = self.create_subscription(
            TimeFloat,
            'jeff_1',
            self.subscription_callback,
            10  # Queue size
        )

        # 发布平面和法向量的 Publisher
        self.publisher_1 = self.create_publisher(
            Marker, 
            'pub_plane_1', 
            10)
        self.normal_publisher_1 = self.create_publisher(
            Marker, 
            'pub_arrow_1', 
            10)
        self.publisher_2 = self.create_publisher(
            Marker, 
            'pub_plane_2', 
            10)
        self.normal_publisher_2 = self.create_publisher(
            Marker, 
            'pub_arrow_2', 
            10)

        self.normal = None
        self.point = None
        self.matrix1 = []
        self.matrix2 = []

        # 创建 1 秒间隔的定时器
        self.timer = self.create_timer(1.0, self.timer_callback)  # 每 1 秒触发一次
  

    def subscription_callback(self, msg):
        """ 解析 Float32MultiArray 并还原两个矩阵 """
        # 读取矩阵 1 形状
        m1_rows = msg.matrix.layout.dim[0].size
        m1_cols = msg.matrix.layout.dim[1].size
        # 读取矩阵 2 形状
        m2_rows = msg.matrix.layout.dim[2].size
        m2_cols = msg.matrix.layout.dim[3].size

        # 解析数据
        flat_data = np.array(msg.matrix.data, dtype=np.float32)
        matrix1 = flat_data[:m1_rows * m1_cols].reshape(m1_rows, m1_cols)
        matrix2 = flat_data[m1_rows * m1_cols:].reshape(m2_rows, m2_cols)

        # 拟合平面
        self.matrix1 = Pixel2World(matrix1 , intrinsic)
        self.matrix2 = Pixel2World(matrix2 , intrinsic)
        plane_fitter_1 = PlaneFitter(self.matrix1)
        self.point_1, self.normal_1 = plane_fitter_1.fit_plane()
        plane_fitter_2 = PlaneFitter(self.matrix2)
        self.point_2, self.normal_2 = plane_fitter_2.fit_plane()

        self.get_logger().info(f"matrix1 is {self.matrix1}")
        self.get_logger().info(f"matrix1 is {self.matrix2}")

        ########### make normal vector always front ##########
        if self.normal_1[0] > 0:
            self.normal_1 = -self.normal_1
        if self.normal_2[0] > 0:
            self.normal_2 = -self.normal_2


    def timer_callback(self):
        """ 发布平面 Marker """
        if self.normal_1 is None or self.point_1 is None:
            return
        
        vertices_1 = compute_plane_vertices(self.normal_1, self.point_1)
        vertices_2 = compute_plane_vertices(self.normal_2, self.point_2)

        ########################################
        ############ publish plane #############
        ### 创建平面 Marker 1
        plane_marker_1 = Marker()
        plane_marker_1.header.frame_id = "camera_link"
        plane_marker_1.header.stamp = self.get_clock().now().to_msg()
        plane_marker_1.ns = "plane"
        plane_marker_1.id = 0
        plane_marker_1.type = Marker.TRIANGLE_LIST
        plane_marker_1.action = Marker.ADD
        plane_marker_1.scale.x = 1.0
        plane_marker_1.scale.y = 1.0
        plane_marker_1.scale.z = 1.0
        plane_marker_1.pose.orientation.w = 1.0  # 无旋转

        # 颜色（RGBA）
        plane_marker_1.color.r = 1.0
        plane_marker_1.color.g = 0.0
        plane_marker_1.color.b = 0.0
        plane_marker_1.color.a = 0.5  # 半透明

        proj_point = project_points_to_plane(self.matrix1, self.point_1, self.normal_1)



        # # 组成两个三角形
        # for i in [0, 2, 1, 0, 3, 2]:
        #     p = Point()
        #     p.x, p.y, p.z = vertices_1[i]
        #     plane_marker_1.points.append(p)

        # show the actural area
        for point in proj_point:
            p = Point()
            p.x = float(point[0])  # x 坐标（米）
            p.y = float(point[1])  # y 坐标（米）
            p.z = float(point[2])  # z 坐标（米）
            plane_marker_1.points.append(p)

        # 关闭平面（如果点足够多，连接首尾形成闭合平面）
        if len(proj_point) > 2:  # 确保至少有 3 个点
            first_point = Point()
            first_point.x = float(proj_point[0, 0])
            first_point.y = float(proj_point[0, 1])
            first_point.z = float(proj_point[0, 2])
            plane_marker_1.points.append(first_point)  # 连接到起点，形成闭合平面
        
        self.publisher_1.publish(plane_marker_1)

        ### 创建平面 Marker 2
        plane_marker_2 = Marker()
        plane_marker_2.header.frame_id = "camera_link"
        plane_marker_2.header.stamp = self.get_clock().now().to_msg()
        plane_marker_2.ns = "plane"
        plane_marker_2.id = 0
        plane_marker_2.type = Marker.TRIANGLE_LIST
        plane_marker_2.action = Marker.ADD
        plane_marker_2.scale.x = 1.0
        plane_marker_2.scale.y = 1.0
        plane_marker_2.scale.z = 1.0
        plane_marker_2.pose.orientation.w = 1.0  # 无旋转

        # 颜色（RGBA）
        plane_marker_2.color.r = 1.0
        plane_marker_2.color.g = 0.0
        plane_marker_2.color.b = 0.0
        plane_marker_2.color.a = 0.5  # 半透明

        # 组成两个三角形
        for i in [0, 2, 1, 0, 3, 2]:
            p = Point()
            p.x, p.y, p.z = vertices_2[i]
            plane_marker_2.points.append(p)

        # 发布平面 Marker
        self.publisher_2.publish(plane_marker_2)

        ########################################
        ############ publish arrow #############
        ### 创建 Marker 1
        normal_maker_1 = Marker()
        normal_maker_1.header.frame_id = "camera_link"
        normal_maker_1.header.stamp = self.get_clock().now().to_msg()
        normal_maker_1.ns = "normal_vector"
        normal_maker_1.id = 1
        normal_maker_1.type = Marker.ARROW
        normal_maker_1.action = Marker.ADD

        # 设置箭头的起点和终点
        start_point_1 = Point()
        start_point_1.x = float(self.point_1[0])
        start_point_1.y = float(self.point_1[1])
        start_point_1.z = float(self.point_1[2])
        end_point_1 = Point()
        end_point_1.x = self.point_1[0] + self.normal_1[0] * 0.5  # 法向量长度缩放
        end_point_1.y = self.point_1[1] + self.normal_1[1] * 0.5
        end_point_1.z = self.point_1[2] + self.normal_1[2] * 0.5

        normal_maker_1.points.append(start_point_1)
        normal_maker_1.points.append(end_point_1)

        # 设置箭头的尺寸和颜色
        normal_maker_1.scale.x = 0.05  # 箭头的宽度
        normal_maker_1.scale.y = 0.1   # 箭头头部宽度
        normal_maker_1.scale.z = 0.1   # 箭头头部高度
        normal_maker_1.color.r = 0.0
        normal_maker_1.color.g = 1.0
        normal_maker_1.color.b = 0.0
        normal_maker_1.color.a = 1.0  # 不透明

        # 发布法向量 Marker
        self.normal_publisher_1.publish(normal_maker_1)

        ### 创建 Marker 2
        normal_maker_2 = Marker()
        normal_maker_2.header.frame_id = "camera_link"
        normal_maker_2.header.stamp = self.get_clock().now().to_msg()
        normal_maker_2.ns = "normal_vector"
        normal_maker_2.id = 1
        normal_maker_2.type = Marker.ARROW
        normal_maker_2.action = Marker.ADD

        # 设置箭头的起点和终点
        start_point_2 = Point()
        start_point_2.x = float(self.point_2[0])
        start_point_2.y = float(self.point_2[1])
        start_point_2.z = float(self.point_2[2])
        end_point_2 = Point()
        end_point_2.x = self.point_2[0] + self.normal_2[0] * 0.5  # 法向量长度缩放
        end_point_2.y = self.point_2[1] + self.normal_2[1] * 0.5
        end_point_2.z = self.point_2[2] + self.normal_2[2] * 0.5

        normal_maker_2.points.append(start_point_2)
        normal_maker_2.points.append(end_point_2)

        # 设置箭头的尺寸和颜色
        normal_maker_2.scale.x = 0.05  # 箭头的宽度
        normal_maker_2.scale.y = 0.1   # 箭头头部宽度
        normal_maker_2.scale.z = 0.1   # 箭头头部高度
        normal_maker_2.color.r = 0.0
        normal_maker_2.color.g = 1.0
        normal_maker_2.color.b = 0.0
        normal_maker_2.color.a = 1.0  # 不透明

        # 发布法向量 Marker
        self.normal_publisher_2.publish(normal_maker_2)


def main():
    rclpy.init()
    node = TestGetPlane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
