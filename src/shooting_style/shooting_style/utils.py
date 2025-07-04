import numpy as np

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


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
        pts.append([dep_x, dep_y, dep_z])

    return np.array(pts)

def flw_spk(pose, board):
    pixel_point = pose[0]
    pixel_point = np.vstack([pixel_point, [0, 0, 0]])
    target_point = Pixel2Rviz(pixel_point, intrinsic)

    return target_point, pixel_point

def spk_brd(pose, board):
    pixel_point = np.vstack((pose[0], board))
    pixel_point = np.mean(pixel_point, axis=0)
    pixel_point = np.vstack([pixel_point, [0, 0, 0]])
    target_point = Pixel2Rviz(pixel_point, intrinsic)
    target_point[0][2] -= 0.5

    return target_point, pixel_point

def center(point):
    pixel_point = np.vstack([point, [0, 0, 0]])
    target_point = Pixel2Rviz(pixel_point, intrinsic)

    return target_point, pixel_point

def World2Robot(W_point):
    x, y, z = W_point[0]
    norm = np.linalg.norm([x, y, z])
    cos_theta_x = x / norm
    angle_rad_x = np.arccos(cos_theta_x)  
    angle_deg_x = np.degrees(angle_rad_x)

    cos_theta_y = y / norm
    angle_rad_y = np.arccos(cos_theta_y)  
    angle_deg_y = np.degrees(angle_rad_y)

    if z < 1.5:
        joint_1 = -0.755 + ((angle_deg_x - 53.534) / 75.631) * 1.629
        joint_2 = 0.339
        joint_3 = -0.369
        joint_5 = -0.376 + ((105.999 - angle_deg_y) / 30.118) * 0.607

    elif 1.5 <= z < 2.0:
        joint_1 = -0.719 + ((angle_deg_x - 53.688) / 77.024) * 1.552
        joint_2 = 0.628
        joint_3 = -0.529
        joint_5 = -0.389 + ((100.401 - angle_deg_y) / 21.11) * 0.514

    elif 2.0 <= z < 2.5:
        joint_1 = -0.7 + ((angle_deg_x - 56.058) / 71.509) * 1.4
        joint_2 = 0.713
        joint_3 = -0.87
        joint_5 = -0.3 + ((109.736 - angle_deg_y) / 44.705) * 1.05

    # Not test
    elif 2.5 <= z:
        joint_1 = -0.64 + ((angle_deg_x - 60.058) / 71.509) * 0.8
        joint_2 = 2.105
        joint_3 = -1.828
        joint_5 = -0.4 + ((95.736 - angle_deg_y) / 20.705) * 0.322

    return joint_1, joint_2, joint_3, joint_5

# x, y, z = W_point[0]
# norm = np.linalg.norm([x, y, z])
# cos_theta_x = x / norm
# angle_rad_x = np.arccos(cos_theta_x)   # 角度与 Y 轴之间的夹角
# angle_deg_x = np.degrees(angle_rad_x)
# print(f'x: {angle_deg_x}')

# cos_theta_y = y / norm
# angle_rad_y = np.arccos(cos_theta_y)   # 角度与 Y 轴之间的夹角
# angle_deg_y = np.degrees(angle_rad_y)
# print(f'y: {angle_deg_y}')

# cos_theta_z = z / norm
# angle_rad_z = np.arccos(cos_theta_z)   # 角度与 Y 轴之间的夹角
# angle_deg_z = np.degrees(angle_rad_z)
# print(f'z: {angle_deg_z}')
