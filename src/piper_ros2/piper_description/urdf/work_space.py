import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import meshcat
import meshcat.geometry as g

# === 加载 URDF 模型（含几何信息） ===
urdf_path = "src/piper_ros2/piper_description/urdf/piper_description.urdf"
model_path = "src/piper_ros2/piper_description"  # 包含 meshes 的路径

model, visual_model, collision_model = pin.buildModelsFromUrdf(
    urdf_path,
    package_dirs=[model_path]
)
data, visual_data, collision_data = pin.createDatas(model, visual_model, collision_model)

# === 初始化 Meshcat 可视化器 ===
viz = MeshcatVisualizer(model, visual_model, collision_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# === 获取关节限制范围 ===
joint_limits = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) for i in range(model.nq)]

# === 设置末端 link 名称 ===
end_effector_frame = model.getFrameId("link7")  # 请确认 link6 是否为末端

# === 随机采样并记录位置 ===
samples = 10000
positions = []

for _ in range(samples):
    q = np.array([np.random.uniform(low, high) for (low, high) in joint_limits])
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pos = data.oMf[end_effector_frame].translation
    positions.append(pos)
    print("End-effector pos:", pos)  # 加这行调试

positions = np.array(positions)

# === 可视化点云 ===
# for i, pos in enumerate(positions):
#     sphere = g.Sphere(0.005)
#     tf = np.eye(4)
#     tf[:3, 3] = pos  # 设置点位置
#     viz.viewer[f"workspace/pt_{i}"].set_object(sphere)
#     viz.viewer[f"workspace/pt_{i}"].set_transform(tf)


viz.viewer["test_point"].set_object(g.Sphere(0.02))
viz.viewer["test_point"].set_transform(np.array([
    [1, 0, 0, 0.3],
    [0, 1, 0, 0.3],
    [0, 0, 1, 0.3],
    [0, 0, 0, 1]
]))
