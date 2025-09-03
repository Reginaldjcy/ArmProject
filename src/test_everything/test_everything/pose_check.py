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
            ## chose more keypoints for better plane fitting
            def line_points(p1, p2, num=10):
                p1, p2 = np.array(p1), np.array(p2)
                ts = np.linspace(0, 1, num)
                return np.array([(1-t)*p1 + t*p2 for t in ts])
                    
            # chest_key = np.vstack((chest_key, line_points(human_1[11], human_1[12], num=5)))
            # chest_key = chest_key[:,:-1]
            # chest_key = get_depth(chest_key, self.depth_image, self.img)

            def line_points(p1, p2, num=10):
                p1, p2 = np.array(p1)[:2], np.array(p2)[:2]   # 只取x,y
                ts = np.linspace(0, 1, num)
                return np.array([(1-t)*p1 + t*p2 for t in ts])

            def points_below_with_depth_filter(
                p1, p2, depth_img,
                num=12,            # 沿两点连线取多少个参考列
                max_down=30,       # 每列向下最多扫描的像素
                abs_tol=0.08,      # 绝对深度容差（米），按你传感器改
                rel_tol=0.12,      # 相对深度容差（比例），两者满足其一即可
                invalid_depth=0    # 无效深度的像素值（有些深度图是0或NaN）
            ):
                h, w = depth_img.shape[:2]
                cols = line_points(p1, p2, num=num)

                valid_pts = []
                for x, y in cols:
                    xi, yi = int(round(x)), int(round(y))
                    if not (0 <= xi < w and 0 <= yi < h):
                        continue

                    z_ref = depth_img[yi, xi]
                    if (np.isnan(z_ref) if np.issubdtype(depth_img.dtype, np.floating) else False) or z_ref == invalid_depth or z_ref <= 0:
                        # 向上寻找替代参考深度
                        found = False
                        for dy in range(1, 6):
                            y_up = yi - dy
                            if y_up < 0: break
                            z_try = depth_img[y_up, xi]
                            if not (np.isnan(z_try) if np.issubdtype(depth_img.dtype, np.floating) else False) and z_try not in (invalid_depth,) and z_try > 0:
                                z_ref = z_try
                                found = True
                                break
                        if not found:
                            continue  # 这一列放弃

                    # 向下扫描
                    for d in range(1, max_down + 1):
                        yj = yi + d
                        if yj >= h: break
                        z = depth_img[yj, xi]
                        if (np.isnan(z) if np.issubdtype(depth_img.dtype, np.floating) else False) or z == invalid_depth or z <= 0:
                            continue

                        # 深度一致性判定
                        if (abs(z - z_ref) <= abs_tol) or (abs(z - z_ref) <= rel_tol * max(z_ref, 1e-6)):
                            valid_pts.append([xi, yj, z])

                return np.array(valid_pts, dtype=float)  # 返回 (x,y,z)
            
            below_pts = points_below_with_depth_filter(
                human_1[11], human_1[12], self.depth_image,
                num=15, max_down=40, abs_tol=0.06, rel_tol=0.10, invalid_depth=0
            )

            # 合并到 chest_key
            if chest_key.shape[1] > 3:
                chest_key = chest_key[:, :3]  # 保留 x,y,z
            if below_pts.size > 0:
                chest_key = np.vstack((chest_key, below_pts))



            # 拟合平面
            chest_rviz= Pixel2Rviz(chest_key, intrinsic)
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
            self.get_logger().info(f'[Pose Plane] 夹角 x: {angles_pose[0]:.2f}°, y: {angles_pose[1]:.2f}°, z: {angles_pose[2]:.2f}°')

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
