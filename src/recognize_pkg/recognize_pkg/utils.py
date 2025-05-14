import numpy as np
import cv2
import pandas as pd
from scipy.spatial import ConvexHull
import time
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from matplotlib.path import Path
from sklearn.decomposition import PCA

# create a board 
# board_size = [x_min, x_max, y_min, y_max]
def CreateBoard(color_image, board_size, z_depth = 3740, color_1=(0, 0, 255)):
    # 矩形左侧区域的 x 值范围（0到20）和 y 值范围（0到100）
    x_range = (board_size[0], board_size[1])
    y_range = (board_size[2], board_size[3])

    # 随机生成 的 x 和 y 坐标
    x_values = np.random.uniform(x_range[0], x_range[1], 100)
    y_values = np.random.uniform(y_range[0], y_range[1], 100)
    z_values = np.full(x_values.shape, z_depth)

    # 将 x 和 y 合并成一个 Nx2 的数组
    points = np.column_stack((x_values, y_values, z_values))
    board_boundary = np.array(points, dtype=np.float32)

    for point in board_boundary:
        cv2.circle(color_image, (int(point[0]), int(point[1])), radius=5, color=color_1, thickness=-1)

    return color_image, board_boundary

# Get depth data 
def get_depth(point_2d, depth_data, img):
    point_3d = []
    for x, y in point_2d:
        if x < img.shape[1] and y < img.shape[0] :
            z = depth_data[int(y), int(x)]            
        else:
            z = 0
        point_3d.append([x, y, z])
    return np.array(point_3d, dtype=np.float32)

##############################################
########### rviz2 orbbec pointcloud ##########
#
#
#                #
#                #     #
#                #    #  
#                #   #   point.x
#                #  #
#                # #                    
#         ########
#           point.y (middle)
#
#
############################################
# convert pixel coordinate to real world x,y
def Pixel2World(keypoints, intrinsic):
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
        start_point = self.points[0] 
        
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


def create_plane_marker(self, obj_points):
        """创建表示棋盘格的三角面片 Marker"""
        marker = Marker()
        marker.header.frame_id = "camera_link" # 确保 RViz2 的 Fixed Frame 设置一致
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "chessboard_plane"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST  # 使用三角形列表绘制平面
        marker.action = Marker.ADD

        # 设置颜色
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5  # 半透明

        # 设置棋盘格的三角形面片
        h, w = self.chessboard_size  # 行数、列数
        for i in range(h - 1):
            for j in range(w - 1):
                # 计算角点索引
                idx0 = i * w + j
                idx1 = idx0 + 1
                idx2 = (i + 1) * w + j
                idx3 = idx2 + 1

                # 添加两个三角形
                for tri in [(idx0, idx1, idx2), (idx1, idx3, idx2)]:
                    for idx in tri:
                        p = Point()
                        p.x = float(obj_points[idx][0])
                        p.y = float(obj_points[idx][1])
                        p.z = float(obj_points[idx][2])
                        marker.points.append(p)

        return marker

def filter_points_in_mask(points, mask):
    """Filter points that are inside the given mask area."""
    points = np.array(points)  # Ensure points is a NumPy array
    mask = np.array(mask)  # Ensure mask is a NumPy array

    # **? ??????? shape**
    print(f"Points shape: {points.shape}")  # (N, 3) ?
    print(f"Mask shape: {mask.shape}")  # (M, 2) ?

    if points.shape[1] < 2:
        raise ValueError("Error: Points should have at least 2 columns (x, y)")

    if mask.shape[1] != 2:
        raise ValueError("Error: Mask should be an (M, 2) array with (x, y) coordinates")

    path = Path(mask)  # Create a Path object from the mask (polygon vertices)

    # **? ??????????? (N, 2) ??? `x, y`**
    inside_mask = path.contains_points(points[:, :2])  

    print(f"Inside mask count: {np.sum(inside_mask)}")  # ??? mask ????

    return points[inside_mask]  # Return points that are inside the mask

def fit_plane(points):
        """
        用最小二乘法拟合平面方程 ax + by + cz + d = 0
        返回平面的法向量 (a, b, c)
        """
        # 提取 X, Y, Z 坐标
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]

        # 构造 A 矩阵和 B 矩阵 (Ax = B 形式)
        A = np.c_[X, Y, np.ones(X.shape)]  # 矩阵 [x, y, 1]
        B = Z  # 目标值 Z

        # 使用最小二乘法求解 a, b, d（其中 c=-1）
        coef, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        a, b, d = coef
        c = -1  # 由于 z = ax + by + d, 可改写为 ax + by - 1z + d = 0

        # 返回法向量
        normal = np.array([a, b, c])
        normal /= np.linalg.norm(normal)  # 归一化
        return normal

def angle_with_axes(normal):
    """
    计算法向量 normal 与 X、Y、Z 轴的夹角（单位：度）
    """
    angles = np.arccos(normal) * 180 / np.pi  # 计算角度并转换为度
    return {"X-axis": angles[0], "Y-axis": angles[1], "Z-axis": angles[2]}

def calculate_bounding_rectangle(points):
    # 将点集转换为 NumPy 数组
    points = np.array(points)
    
    # 使用 PCA 进行主成分分析，找到数据的主方向
    pca = PCA(n_components=2)  # 只考虑 x 和 y 平面的主成分
    pca.fit(points) 
    
    # 获取主成分方向
    components = pca.components_
    
    # 将所有点投影到主成分上
    transformed_points = pca.transform(points)
    
    # 计算投影后的最大和最小值，得到矩形的长和宽
    width = np.max(transformed_points[:, 0]) - np.min(transformed_points[:, 0])
    height = np.max(transformed_points[:, 1]) - np.min(transformed_points[:, 1])
    
    return height, width


import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def min_bounding_rectangle(points):
    # 计算凸包
    hull = ConvexHull(points)
    hull_points = np.array([points[i] for i in hull.vertices])

    # 使用 Shapely 计算最小外接矩形
    polygon = Polygon(hull_points)
    min_rect = polygon.minimum_rotated_rectangle

    # 获取外接矩形的四个角点坐标
    rect_coords = list(min_rect.exterior.coords)[:-1]  # 最后一个点和第一个点重复

    # 计算每条边的长度
    edge_lengths = [np.linalg.norm(np.array(rect_coords[i]) - np.array(rect_coords[i-1])) for i in range(1, len(rect_coords))]
    # print(f"edge length is {edge_lengths}")

    # 按长度排序，最长的作为"长"，次长的作为"宽"
    width, height = sorted(edge_lengths)[-2:]
    width = max(edge_lengths)
    height = min(edge_lengths)

    return width, height

def shrink_rectangle(pts, shrink_amount=2):
    pts = np.array(pts, dtype=np.float32)
    new_pts = []

    for i in range(4):
        prev = pts[(i - 1) % 4]
        curr = pts[i]
        next = pts[(i + 1) % 4]

        # 两个相邻边向量
        v1 = curr - prev
        v2 = next - curr

        # 单位法向量（垂直方向朝内）
        n1 = np.array([-v1[1], v1[0]])
        n2 = np.array([-v2[1], v2[0]])
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)

        # 两个法向平均
        move_dir = (n1 + n2)
        move_dir /= np.linalg.norm(move_dir)

        new_pt = curr + move_dir * shrink_amount
        new_pts.append(new_pt)

    return np.array(new_pts, dtype=np.int32)

def sort_pts_counterclockwise(pts):
    pts = np.array(pts, dtype=np.float32).reshape((-1, 2))

    # 计算中心
    center = np.mean(pts, axis=0)
    # 计算角度
    def angle(p): return np.arctan2(p[1] - center[1], p[0] - center[0])
    sorted_pts = sorted(pts, key=angle)
    return np.array(sorted_pts, dtype=np.int32)
