import numpy as np
import cv2
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

##############################################
########### rviz2 orbbec pointcloud ##########
#
#                z
#                #
#                #     #
#                #    #  
#                #   #   x
#                #  #
#                # #                    
#         ########
#     y 
#
#
############################################
# convert pixel coordinate to real world x,y
def Pixel2Rviz(keypoints, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    keypoints = np.array(keypoints)
    pts = []

    for point in keypoints:
        x, y, z = float(point[0]), float(point[1]), float(point[2])  # 确保数据为浮点数

        if z == 0:  # 处理深度为 0 的情况
            pts.append([0.0, 0.0, 0.0])
            continue

        dep_x = ((x - cx) * z / fx) / 1000.0        #((y - cy) * z / fy) / 1000.0       
        dep_y = ((y - cy) * z / fy) / 1000.0         #((x - cx) * z / fx) / 1000.0      
        dep_z = z / 1000.0

        # for normal xyz
        # pts.append([dep_x, dep_y, dep_z])
        pts.append([dep_z, -dep_x, -dep_y])

    return np.array(pts)



def Pixel2Optical(keypoints, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    keypoints = np.array(keypoints)
    pts = []

    for point in keypoints:
        x, y, z = float(point[0]), float(point[1]), float(point[2])  # 确保数据为浮点数

        if z == 0:  # 处理深度为 0 的情况
            pts.append([0.0, 0.0, 0.0])
            continue

        dep_x = ((x - cx) * z / fx) / 1000.0       
        dep_y = ((y - cy) * z / fy) / 1000.0         
        dep_z = z / 1000.0

        pts.append([dep_x, dep_y, dep_z])

    return np.array(pts)


# SVD calculate the plane
# calculate the fit plane
class PlaneFitter:
    def __init__(self, points):
        self.points = points
        self.plane_normal = None
        self.centroid = None

    def fit_plane(self):
     
        self.centroid = np.mean(self.points, axis=0)    
        shifted_points = self.points - self.centroid
        _, _, vh = np.linalg.svd(shifted_points)
        self.plane_normal = vh[-1]
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        # detect the start point of normal vector
        start_point = 0.5 * (self.points[0] + self.points[1])
        
        return start_point, self.plane_normal

# calculate points in fixed plane
def compute_plane_vertices(normal, point, size=1.0):
    """ 根据法向量和点计算平面四个顶点 """
    A, B, C = normal

    # 计算平面内的两个方向向量，确保与法向量垂直
    # 我们选择 x 轴或者 y 轴作为起始点，根据法向量的分量来选择合适的 T1 向量
    if A != 0 or B != 0:
        T1 = np.array([-B, A, 0])  # 保证与法向量垂直
    else:
        T1 = np.array([1, 0, 0])  # 如果 A 和 B 都为零，选择 x 轴方向

    T2 = np.cross(normal, T1)  # 计算与法向量垂直的第二个方向向量

    # 归一化方向向量
    T1 = T1 / np.linalg.norm(T1) * size / 2
    T2 = T2 / np.linalg.norm(T2) * size / 2

    # 计算平面四个顶点
    P1 = point + T1 + T2
    P2 = point + T1 - T2
    P3 = point - T1 - T2
    P4 = point - T1 + T2

    return [P1, P2, P3, P4]


def project_points_to_plane(points, centroid, normal):
    # 计算每个点到平面的投影
    projected_points = np.zeros_like(points, dtype=float)
    
    for i, point in enumerate(points):
        # 点到平面的向量
        vector_to_point = point - centroid
        # 点到平面的距离（沿着法向量）
        distance = np.dot(vector_to_point, normal)
        # 投影点 = 原点 - 距离 * 法向量
        projected_points[i] = point - distance * normal
    
    return projected_points


import rclpy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

def create_point_marker(points, frame_id="camera_color_optical_frame", ns="point_cloud", marker_id=0, 
                               scale=0.01, color=(0.0, 1.0, 0.0, 1.0), lifetime_sec=0):
    

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    marker.scale.x = scale
    marker.scale.y = scale

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    marker.points = points
    marker.lifetime = Duration(sec=lifetime_sec)

    return marker

def create_plane_marker(corners, frame_id="camera_color_optical_frame") -> Marker:


    # 检查类型，如果是 list of Point，先转 numpy
    if isinstance(corners[0], Point):
        corners = np.array([[p.x, p.y, p.z] for p in corners])

    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "plane"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST  # 使用三角形列表绘制
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0  # 方向默认不旋转
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.pose.orientation.w = 1.0  # 无旋转

    # 颜色（RGBA）
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 0.5  # 半透明

    # 解析矩阵中的四个点
    p1 = Point(x=corners[0, 0], y=corners[0, 1], z=corners[0, 2])
    p2 = Point(x=corners[1, 0], y=corners[1, 1], z=corners[1, 2])
    p3 = Point(x=corners[2, 0], y=corners[2, 1], z=corners[2, 2])
    p4 = Point(x=corners[3, 0], y=corners[3, 1], z=corners[3, 2])

    # 组成两个三角形
    marker.points.extend([p1, p3, p2, p1, p4, p3])

    return marker

def create_arrow_marker(start_point, normal_vector, frame_id="camera_color_optical_frame", marker_id=1, scale=0.5, color=None) -> Marker:

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "normal_vector"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # 设置箭头起点和终点
    start = Point(x=float(start_point[0]), y=float(start_point[1]), z=float(start_point[2]))
    end = Point(
        x=start.x + normal_vector[0] * scale,
        y=start.y + normal_vector[1] * scale,
        z=start.z + normal_vector[2] * scale
    )
    
    marker.points.append(start)
    marker.points.append(end)

    # 设置箭头的尺寸
    marker.scale.x = 0.05  # 箭杆宽度
    marker.scale.y = 0.1   # 箭头头部宽度
    marker.scale.z = 0.1   # 箭头头部高度

    # 设置颜色（默认绿色）
    if color is None:
        color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # blue
    marker.color = color

    return marker

def pixel_to_robot_frame(point, diff):
    """
    Convert pixel coordinates to robot frame coordinates.
    
    Args:
        point (np.array): Pixel coordinates in the form [x, y, z].
        diff (np.array): Difference vector to be added to the pixel coordinates.
        
    Returns:
        np.array: Robot frame coordinates.
    """
    # Ensure point and diff are numpy arrays
    point = np.array(point[0], dtype=float)
    diff = np.array(diff, dtype=float)

    # Convert pixel coordinates to robot frame
    robot_frame = np.array([point[0] - diff[0], point[1] + diff[1], point[2] + diff[2]])

    return robot_frame

import numpy as np

def how2shoot(face_center, normal_vector, face_dist, robot_position, robot_radius):

    # Step 1: Try 0.5m forward along normal
    target_point = face_center + face_dist * normal_vector
    target_to_robot = np.linalg.norm(target_point - robot_position)

    if target_to_robot <= robot_radius:
        print('With in robot sphere')
        return target_point
    else:
        # Step 2: Ray from face_center toward robot_position, find intersection with sphere
        face_to_robot = np.linalg.norm(face_center - robot_position)
        if face_to_robot <= face_dist + robot_radius:
            print('In face sphere')
            direction_1 = target_point - robot_position
            unit_dir_1 = direction_1 / np.linalg.norm(direction_1)
            intersection_1 = robot_position + robot_radius * unit_dir_1
            return intersection_1
        else:
            print('Very far away')
            # Step 3: Find intersection with sphere
            direction_2 = face_center - robot_position
            unit_dir_2 = direction_2 / np.linalg.norm(direction_2)
            intersection_2 = robot_position + robot_radius * unit_dir_2
            return intersection_2

def get_depth(point_2d, depth_data, img):
    point_3d = []
    for x, y in point_2d:
        if x < img.shape[1] and y < img.shape[0] :
            z = depth_data[int(y), int(x)]            
        else:
            z = 0
        point_3d.append([x, y, z])
    return np.array(point_3d, dtype=np.float32)

def chest_points_vertical(
    p1, p2, p3, p4, depth_img,
    num=50,            # 在 p1–p2 连线采样的点数
    max_down=600,      # 每列向下扫描的最大像素数
    step=5,            # 扫描步长
    abs_tol=0.1,      # 绝对深度容差
    rel_tol=0.15,      # 相对深度容差
    invalid_depth=0    # 无效深度值
):
    
    valid_pts = []

    if p3[2] <100000 or p4[2]<10000:
        # not detetct hip
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
    else:
        # detetct hip
        valid_pts = np.vstack((p3, p4))
        

    return np.array(valid_pts, dtype=float)
