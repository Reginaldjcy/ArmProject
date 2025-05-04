import numpy as np
import cv2
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Determine the distance between speaker and board 
# human_1 (x, y, z)
# board_boundary (x, y, z)
def speaker_board_left_right(human_1, board_boundary, frame):
    # mean x value of human
    selected_human_xy = human_1[:, :2]

    # selected_human_xy = np.array(selected_human_xy)
    #print(selected_human_xy)

    if selected_human_xy.size == 0:
        x_speaker_mean = 0
    else:
        x_speaker_mean = np.mean(selected_human_xy[:, 0])

    ######################## calculate outrange of board #########################################
    # outrange rentangle 
    board_boundary_2d = board_boundary[:, :2]

    hull = ConvexHull(board_boundary_2d)  
    hull_points = board_boundary_2d[hull.vertices]  # 获取凸包上的点


    # 使用 OpenCV 函数找到最小外接矩形
    rect = cv2.minAreaRect(hull_points)  
    box = cv2.boxPoints(rect)  # 获取矩形的四个角点
    box = np.int32(box)  # 转换为整数

    cv2.polylines(frame, [box], isClosed=True, color=(0, 255, 255), thickness=2)

    box = box[:, 0]
    box = np.sort(box, axis=0)
    #########################################################################

    # check left and right boundary of board
    board_boundary_left = np.mean(box[:2])
    board_boundary_right = np.mean(box[-2:])

    # determine speaker left or right
    if x_speaker_mean < board_boundary_left:
        relative_position = "left"
        
    elif x_speaker_mean > board_boundary_right:
        relative_position = "right"
        
    else:
        relative_position = "inside"
    
    return frame, relative_position


# Speaker and board close board distance
# board_boundary (x, y, z)
# selected_human_xyz = (x, y, z)
def speaker_board_dist(color_image, relative_position, human_1, board_boundary, percentile = 5, coef = 1e8,):

    # print(f"input human is {selected_human_xy}")

    ############################################################################
    #  boundary for board 
    left_x_threshold = np.percentile(board_boundary[:, 0], percentile)  
    right_x_threshold = np.percentile(board_boundary[:, 0], (100-percentile))  

    left_board_3d = board_boundary[board_boundary[:, 0] <= left_x_threshold]
    right_board_3d = board_boundary[board_boundary[:, 0] >= right_x_threshold]

    ###########################################################################
    #  boundary for human
    if human_1.size == 0:
        x_max = x_min = y_max = y_min = depth_mean = 0  
        points = np.array([[1.1, 1.1, 1.1], [1.1, 1.1, 1.1]])  

    else:
        points = human_1
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        depth_mean = np.mean(points[:, 2])

    #####################
    # depth_mean = 3400 #
    #####################

    left_human_3d =  np.array([[x_min, y_min, depth_mean], [x_min, y_max, depth_mean]])
    right_human_3d = np.array([[x_max, y_min, depth_mean], [x_max, y_max, depth_mean]])

    human_box = np.array([left_human_3d[0][:2], left_human_3d[1][:2], right_human_3d[1][:2], right_human_3d[0][:2]], dtype=np.int32)

    cv2.polylines(color_image, [human_box], isClosed=True, color=(0, 255, 0), thickness=2)
    #print(f"mean human depth is {depth_mean}")

    ############################################################################
    # choose which side 
    if relative_position == "left":

        relative_dist = (abs(np.mean(left_board_3d[:, 0]) - np.mean(right_human_3d[:, 0])) 
                         / coef * np.mean(np.mean(left_board_3d[:, 2]) + np.mean(right_human_3d[:, 2])))

        # draw board
        x1, y1 = left_board_3d[0, :2]
        x2, y2 = left_board_3d[1, :2]
        cv2.line(color_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(225, 0, 255), thickness=2)

        # draw human
        cv2.polylines(color_image, [human_box[2:4]], isClosed=True, color=(225, 0, 255), thickness=2)

        #print(f"upper is : {(np.mean(left_board_3d[:, 0]) - np.mean(right_human_3d[:, 0])) }")
        #print(f"down is : {np.mean(np.mean(left_board_3d[:, 2]) + np.mean(right_human_3d[:, 2]))}")


    elif relative_position == "right":
        
        relative_dist = (abs(np.mean(right_board_3d[:, 0]) - np.mean(left_human_3d[:, 0])) 
                    / coef * np.mean(np.mean(right_board_3d[:, 2]) + np.mean(left_human_3d[:, 2])))
        
        # draw board
        x1, y1 = right_board_3d[0, :2]
        x2, y2 = right_board_3d[1, :2]
        cv2.line(color_image, (int(x1), int(y1)), (int(x2), int(y2)), color=(225, 0, 255), thickness=2)
        
        # draw human
        cv2.polylines(color_image, [human_box[0:2]], isClosed=True, color=(225, 0, 255), thickness=2) 

    else:
        relative_dist = np.array([1e-6])


    return relative_dist, color_image

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

# determain whether have intersect with board
def ray_intersects_plane(ray_origin, ray_direction, plane_points):
    """
    判断射线是否与由四个点构成的矩形平面相交
    
    参数:
    ray_origin: tuple (x, y, z) - 射线起点
    ray_direction: tuple (x, y, z) - 射线方向
    plane_points: list of 4 tuples [(x1,y1,z1), (x2,y2,z2), ...] - 平面四个顶点（顺时针或逆时针）
    
    返回:
    bool - 是否相交
    tuple - 交点坐标 (如果相交) 或 None
    """
    # 转换为numpy数组
    origin = np.array(ray_origin)
    direction = np.array(ray_direction) / np.linalg.norm(ray_direction)
    points = np.array(plane_points)
    
    # 计算平面法向量
    v1 = points[1] - points[0]
    v2 = points[2] - points[1]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # 检查是否平行
    denominator = np.dot(direction, normal)
    if abs(denominator) < 1e-6:
        return False, None
    
    # 计算交点参数t
    w = origin - points[0]
    t = -np.dot(w, normal) / denominator
    
    # 交点在射线后面
    if t < 0:
        return False, None
    
    # 计算交点
    intersection = origin + t * direction
    
    # 检查交点是否在四边形内
    # 将问题转换为2D投影，计算参数坐标
    # 选择两个基向量（边的方向）
    u_vec = (points[1] - points[0])  # 边0->1
    v_vec = (points[3] - points[0])  # 边0->3 （假设矩形顺序）
    u_vec = u_vec / np.linalg.norm(u_vec)
    v_vec = v_vec / np.linalg.norm(v_vec)
    
    # 计算交点在局部坐标系中的参数
    diff = intersection - points[0]
    u = np.dot(diff, u_vec)
    v = np.dot(diff, v_vec)
    
    # 计算边长
    u_length = np.linalg.norm(points[1] - points[0])
    v_length = np.linalg.norm(points[3] - points[0])
    
    # 检查是否在矩形范围内
    if (0 <= u <= u_length) and (0 <= v <= v_length):
        return True, tuple(intersection)
    return False, None

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
from builtin_interfaces.msg import Time
from builtin_interfaces.msg import Duration
import rclpy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration

def create_hopenet_marker(x, y, z, quat, frame_id="camera_link", marker_id=0, timestamp=None):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = timestamp if timestamp else Time()
    marker.ns = "arrows"
    marker.id = marker_id
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

    return marker


def create_point_marker(position, frame_id="camera_link", marker_id=1):
    
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "intersect_point"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    # 设置位置
    marker.pose.position = Point(x=position[0], y=position[1], z=position[2])

    # 设置默认方向（四元数必须设一个单位值）
    marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    # 设置大小
    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    # 设置颜色（蓝色）
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    return marker

def create_arrow_marker(start_point, normal_vector, frame_id="camera_link", marker_id=1, scale=0.5, color=None) -> Marker:

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
        color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # 绿色
    marker.color = color

    return marker