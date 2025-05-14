# calibrate_camera.py
import cv2
import numpy as np
import glob
import yaml

# 棋盘格参数
CHECKERBOARD = (8, 6)
SQUARE_SIZE = 0.025  # 单位：米

# 构建世界坐标点
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3d 点
imgpoints = []  # 2d 点

images = glob.glob('calib_imgs/*.png')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(200)

cv2.destroyAllWindows()

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix:\n", K)
print("Distortion coefficients:\n", dist)

# 保存为 YAML
data = {
    'camera_matrix': {'rows': 3, 'cols': 3, 'data': K.flatten().tolist()},
    'distortion_coefficients': {'data': dist.flatten().tolist()}
}
with open("camera.yaml", "w") as f:
    yaml.dump(data, f)
