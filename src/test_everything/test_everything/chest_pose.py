import rclpy
from rclpy.node import Node

from msg_interfaces.msg import TimeFloat
from message_filters import Subscriber, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image

import numpy as np
from .utils import *
from cv_bridge import CvBridge
import cv2

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


class TestGetPlane(Node):
    def __init__(self):
        super().__init__('publish_pose_brd')

        # Subscription
        self.pose_sub = self.create_subscription(TimeFloat, 'pose_1', self.subscription_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.rgb_callback, 10)

        # 发布 arrow
        self.pose_arrow = self.create_publisher(Marker, 'pose_arrow', 10)
        self.select_points = self.create_publisher(Marker, 'select_points', 10)

        # CV2 bridge
        self.bridge = CvBridge()
        self.depth_image = None
        self.img = None 

    def depth_callback(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    def rgb_callback(self, rgb_msg):
        self.img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
  
    def subscription_callback(self, msg):
        human_1 = np.array(msg.matrix.data).reshape(-1,3)
        ## chose human body 
        chest_key = human_1[[11,12]]
        if self.depth_image is not None or self.img is not None:
            # 根据胸部关键点，向下扫描，获取更多胸部区域点
            def chest_points_vertical(
                p1, p2, depth_img,
                num=50,            # 在 p1–p2 连线采样的点数
                max_down=600,      # 每列向下扫描的最大像素数
                step=5,            # 扫描步长
                abs_tol=0.1,      # 绝对深度容差
                rel_tol=0.15,      # 相对深度容差
                invalid_depth=0    # 无效深度值
            ):
                """
                p1, p2: (x,y,z) 带深度的胸部关键点
                depth_img: 深度图
                返回:
                valid_pts: (N,3)，胸部点 (x,y,z)
                """
                h, w = depth_img.shape[:2]
                x1, y1, z1 = p1
                x2, y2, z2 = p2

                # 深度区间
                z_min, z_max = min(z1, z2), max(z1, z2)
                band_low  = z_min - abs_tol
                band_high = z_max + abs_tol
                base_scale = max(z1, z2, 1e-6)

                # 在 p1–p2 连线上采样
                xs = np.linspace(x1, x2, num)
                ys = np.linspace(y1, y2, num)
                cols = np.stack([xs, ys], axis=1)

                valid_pts = []

                # 遍历每个采样点，竖直向下扫描
                for x, y in cols:
                    xi, yi = int(round(x)), int(round(y))
                    if not (0 <= xi < w and 0 <= yi < h):
                        continue

                    for d in range(step, max_down+1, step):
                        xj = xi
                        yj = yi + d
                        if not (0 <= xj < w and 0 <= yj < h):
                            break

                        z = depth_img[yj, xj]
                        if np.issubdtype(depth_img.dtype, np.floating) and np.isnan(z):
                            continue
                        if z == invalid_depth or z <= 0:
                            continue
                        z = float(z)

                        # 深度条件
                        in_band = (band_low <= z <= band_high)
                        close_to_any = (abs(z - z1) <= rel_tol*base_scale) or \
                                    (abs(z - z2) <= rel_tol*base_scale)

                        if in_band or close_to_any:
                            valid_pts.append([xj, yj, z])

                return np.array(valid_pts, dtype=float)
            
            below_pts = chest_points_vertical(human_1[11], human_1[12], self.depth_image,)

            # 合并到 chest_key
            if chest_key.shape[1] > 3:
                chest_key = chest_key[:, :3]  # 保留 x,y,z
            if below_pts.size > 0:
                    if below_pts.shape[0] > 50:
                        idx = np.random.choice(below_pts.shape[0], 10, replace=False)  # 随机选择10个
                        below_pts_sample = below_pts[idx]
                    else:
                        below_pts_sample = below_pts
                    chest_key = np.vstack((chest_key, below_pts_sample))
            self.get_logger().info(f"chest keypoints num: {chest_key.shape[0]}")

            chest_rviz= Pixel2Rviz(chest_key, intrinsic)

            # 发布选择点
            select_points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in chest_rviz]
            select_points = create_point_marker(select_points)
            self.select_points.publish(select_points)


            # 拟合平面
            plane_fitter_1 = PlaneFitter(chest_rviz)
            point_pose, normal_pose = plane_fitter_1.fit_plane()


            ########### make normal vector always front ##########
            if normal_pose[0] > 0:
                normal_pose = -normal_pose

            ########## for arrow and plane publish #########
            if normal_pose is None:
                return
            
            # 计算法向量与坐标轴夹角（单位：度）
            def compute_angles(normal):
                norm = np.linalg.norm(normal)
                angle_x = np.arccos(normal[0] / norm) * 180 / np.pi
                angle_y = np.arccos(normal[1] / norm) * 180 / np.pi
                angle_z = np.arccos(normal[2] / norm) * 180 / np.pi
                return angle_x, angle_y, angle_z

            angles_pose = compute_angles(normal_pose)


            # 打印结果
            # self.get_logger().info(f'[Pose Plane] 夹角 x: {angles_pose[0]:.2f}°, y: {angles_pose[1]:.2f}°, z: {angles_pose[2]:.2f}°')

            ### publish arrow
            pose_arrow = create_arrow_marker(point_pose, normal_pose)

            self.pose_arrow.publish(pose_arrow)



def main():
    rclpy.init()
    node = TestGetPlane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
