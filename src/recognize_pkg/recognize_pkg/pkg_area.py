import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from msg_interfaces.msg import TimeFloat
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import Polygon, Point32

from cv_bridge import CvBridge
import cv2
import numpy as np
from .utils import *

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])

class PubArea(Node):
    def __init__(self):
        super().__init__('pub_area')
        
        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        # pub
        self.obj_pub = self.create_publisher(Float32MultiArray, '/obj_point', 10)
        self.bot_pub = self.create_publisher(Float32MultiArray, '/bot_point', 10)
        self.all_pub = self.create_publisher(Float32MultiArray, '/all_point', 10)

        self.bridge = CvBridge()

    def sync_callback(self, rgb_msg, depth_msg):
        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough') 

        # 2D show select area
        if True:
            # preset area
            area = np.array([
                [260, 120],
                [820, 120],
                [820, 520],
                [260, 520]
            ])     
            area_int = area.astype(np.int32)
            cv2.polylines(color_image, [area_int], isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('Rectangle', color_image)
        cv2.waitKey(1)  # 让 OpenCV 正常刷新图像窗口

        #########################
        ##### get all points ####
        # 获取矩形的整数边界
        x_min, y_min = np.floor(np.min(area, axis=0)).astype(int)
        x_max, y_max = np.ceil(np.max(area, axis=0)).astype(int)

        # 生成网格点
        x_vals = np.arange(x_min, x_max + 1)  # 所有 x 坐标
        y_vals = np.arange(y_min, y_max + 1)  # 所有 y 坐标
        X, Y = np.meshgrid(x_vals, y_vals)  # 生成网格

        # 将网格点转换为 (N, 2) 形式的整数点列表
        integer_points = np.column_stack((X.ravel(), Y.ravel()))

        ####################
        #### world data ####
        integer_depth = get_depth(integer_points, depth_image, color_image)
        integer_world = Pixel2Rviz(integer_depth, intrinsic)

        # 过滤掉 z==0 的点
        integer_world = integer_world[integer_world[:, 2] != 0]

        # filter to get upper point
        plane1 = []
        plane2 = []
        x_threshold1 = 0.5
        x_threshold2 = 0.600
        x_threshold3 = 0.698

        for point in integer_world:
            x = point[0]
            if x_threshold2 < x < x_threshold3:
                plane1.append(point)
                
            elif x > x_threshold3:
                plane2.append(point)

        plane1 = np.array(plane1)
        plane2 = np.array(plane2)

        #########################
        ##### Publish point #####
        integer_world = integer_world.astype(np.float32)
        plane1 = plane1.astype(np.float32)
        plane2 = plane2.astype(np.float32)

        all_marker = Float32MultiArray()
        obj_marker = Float32MultiArray()
        bot_marker = Float32MultiArray()

        all_marker.data = integer_world.flatten().tolist()
        obj_marker.data = plane1.flatten().tolist()
        bot_marker.data = plane2.flatten().tolist()

        self.all_pub.publish(all_marker)
        self.obj_pub.publish(obj_marker)
        self.bot_pub.publish(bot_marker)

def main(args=None):
    rclpy.init(args=args)
    node = PubArea()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()













# class ChessboardPlaneDetector(Node):
#     def __init__(self):
#         super().__init__('chessboard_plane_detector')

#         # Subscribers for RGB and Depth images
#         self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
#         self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

#         # Synchronize RGB and Depth images
#         self.sync = ApproximateTimeSynchronizer(
#             [self.rgb_sub, self.depth_sub],
#             queue_size=10,
#             slop=0.1
#         )
#         self.sync.registerCallback(self.sync_callback)

#         # pub
#         self.marker_pub = self.create_publisher(Marker, '/chessboard_plane_marker', 10)
#         self.polygon_pub = self.create_publisher(Polygon, '/chessboard_plane', 10)
                
#         # 棋盘格大小 (内角点)
#         self.chessboard_size = (8, 6)  
#         self.square_size = 0.025  # 假设每个格子 25mm（单位：米）
#         self.bridge = CvBridge()

#         # 相机内参（需要根据实际相机标定）
#         self.camera_matrix = np.array([[688.4984130859375, 0.0, 639.0274047851562],
#                                        [0.0, 688.466552734375, 355.8525390625],
#                                        [0.0, 0.0, 1.0]])

#     def sync_callback(self, rgb_msg, depth_msg):
#         # Convert ROS Image messages to OpenCV images
#         color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
#         depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

#         gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

#         ret, corners = cv2.findChessboardCorners(
#             gray, self.chessboard_size, 
#             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
#         )

#         if True:
#             # 在图像上绘制棋盘角点
#             cv2.drawChessboardCorners(color_image, self.chessboard_size, corners, ret)
#             corners = corners.reshape(-1, 2)
#             corners = get_depth(corners, depth_image, color_image)
#             matrix1 = Pixel2Rviz(corners , self.camera_matrix)

#             # marker = create_plane_marker(self, points)
#             # self.marker_pub.publish(marker)

#             self.polygon_pub.publish(self.polygon_msg)
#             self.polygon_to_marker
#             self.get_logger().info("publish cheeseboard plane")

#         # 展示图像
#         cv2.imshow("Chessboard Detection", color_image)
#         cv2.waitKey(1)  # 让 OpenCV 正常刷新图像窗口

# def main(args=None):
#     rclpy.init(args=args)
#     node = ChessboardPlaneDetector()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
