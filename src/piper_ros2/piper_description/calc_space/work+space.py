import pinocchio as pin
from os.path import join

# 路径设置
model_path = "/home/reginald/ArmProject/install/piper_description/share/piper_description"
urdf_filename = "urdf/piper_description.urdf"
urdf_path = join(model_path, urdf_filename)

# 加载模型
model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()

# 可选：加载视觉几何等（如果你要可视化或做碰撞检测）
# model, visual_model, collision_model = pin.buildModelsFromUrdf(urdf_path)

# for i, frame in enumerate(model.frames):
#     print(i, frame.name)

ee_frame_name = "link6"  # 或 "gripper_base" 视情况而定
ee_frame_id = model.getFrameId(ee_frame_name)

import numpy as np

workspace_points = []
n_samples = 10000

for _ in range(n_samples):
    q = pin.randomConfiguration(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacement(model, data, ee_frame_id)
    pos = data.oMf[ee_frame_id].translation
    workspace_points.append(pos)

workspace_points = np.array(workspace_points)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(workspace_points[:,0], workspace_points[:,1], workspace_points[:,2], s=1, alpha=0.3)
ax.set_title("Agilex PIPER Workspace (Pinocchio)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()