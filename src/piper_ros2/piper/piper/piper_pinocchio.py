#!/usr/bin/env python3
import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
import math
import pinocchio.casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import threading
from .piper_control_ros2 import PIPER
from scipy.spatial.transform import Rotation as R  # 用于欧拉角到四元数的转换
# ROS 2 core packages
import rclpy
from rclpy.node import Node

# TF2 packages in ROS 2
import tf2_ros

# ROS 2 message types
from std_msgs.msg import Float64, Float32MultiArray
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import Pose, PoseStamped


from ament_index_python.packages import get_package_share_directory
from .utils import *


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class Arm_IK:
    def __init__(self):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        # urdf_path = '/home/agilex/piper_ws/src/piper_description/urdf/piper_description.urdf'
        # urdf_path = rospkg.RosPack().get_path('piper_description') + '/urdf/piper_description.urdf'
        self.piper_node = PIPER()
        urdf_path = os.path.join(get_package_share_directory('piper_description'), 'urdf/piper_description.urdf')
        self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )
        q = R.from_euler('xyz', [0, -math.pi/2, 0]).as_quat()
        link6_id= self.reduced_robot.model.getFrameId("link6")
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.frames[link6_id].parentJoint,
                        link6_id,
                      pin.SE3(
                          # pin.Quaternion(1, 0, 0, 0),
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([0.0, 0.0, 0.0]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

        self.geom_model = pin.buildGeomFromUrdf(self.robot.model, urdf_path, pin.GeometryType.COLLISION)
        for i in range(4, 9):
            for j in range(0, 3):
                self.geom_model.addCollisionPair(pin.CollisionPair(i, j))
        self.geometry_data = pin.GeometryData(self.geom_model)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.history_data = np.zeros(self.reduced_robot.model.nq)

        # # Initialize the Meshcat visualizer  for visualization
        self.vis = MeshcatVisualizer(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        self.vis.initViewer(open=True)
        self.vis.loadViewerModel("pinocchio")
        self.vis.displayFrames(True, frame_ids=[113, 114], axis_length=0.15, axis_width=5)
        self.vis.display(pin.neutral(self.robot.model))
        # self.joint_state_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        # Enable the display of end effector target frames with short axis lengths and greater width.
        frame_viz_names = ['ee_target']
        FRAME_AXIS_POSITIONS = (
            np.array([[0, 0, 0], [1, 0, 0],
                      [0, 0, 0], [0, 1, 0],
                      [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
        )
        FRAME_AXIS_COLORS = (
            np.array([[1, 0, 0], [1, 0.6, 0],
                      [0, 1, 0], [0.6, 1, 0],
                      [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
        )
        axis_length = 0.1
        axis_width = 10
        # for frame_viz_name in frame_viz_names:
        #     self.vis.viewer[frame_viz_name].set_object(
        #         mg.LineSegments(
        #             mg.PointsGeometry(
        #                 position=axis_length * FRAME_AXIS_POSITIONS,
        #                 color=FRAME_AXIS_COLORS,
        #             ),
        #             mg.LineBasicMaterial(
        #                 linewidth=axis_width,
        #                 vertexColors=True,
        #             ),
        #         )
        #     )

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf = casadi.SX.sym("tf", 4, 4)
        self.cR = casadi.SX.sym("r", 3, 3)
        self.cT = casadi.SX.sym("t", 3, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)
        
        # Define weights
        rot_weight = 1.0   # weight for rotation error
        pos_weight = 10.0  # weight for position error

        # Create weight matrix (6x6 diagonal)
        W = casadi.diag(casadi.SX([rot_weight]*2 + [pos_weight]*4))

        # # Get the hand joint ID and define the error function
        self.gripper_id = self.reduced_robot.model.getFrameId("ee")
        self.error = casadi.Function(
            "error",
            [self.cq, self.cTf],
            [
                casadi.vertcat(
                    cpin.log6(
                        self.cdata.oMf[self.gripper_id].inverse() * cpin.SE3(self.cTf)
                    ).vector
                )
            ],
        )

        

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)   # for smooth
        self.param_tf = self.opti.parameter(4, 4)
        self.totalcost = casadi.sumsqr(self.error(self.var_q, self.param_tf))
        self.regularization = casadi.sumsqr(self.var_q)
        # self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last) # for smooth

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.reduced_robot.model.lowerPositionLimit,
            self.var_q,
            self.reduced_robot.model.upperPositionLimit)
        )
        # print("self.reduced_robot.model.lowerPositionLimit:", self.reduced_robot.model.lowerPositionLimit)
        # print("self.reduced_robot.model.upperPositionLimit:", self.reduced_robot.model.upperPositionLimit)
        self.opti.minimize(10 * self.totalcost +0.1* self.regularization )
        # self.opti.minimize(20 * self.totalcost + 0.01 * self.regularization + 0.1 * self.smooth_cost) # for smooth

        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 50,
                'tol': 1e-4,
            },
            'print_time': False
        }
        self.opti.solver("ipopt", opts)
        # self.opti.solver("sqpmethod")
        
        

    def ik_fun(self, target_pose, gripper=0, motorstate=None, motorV=None):
        gripper = np.array([gripper/2.0, -gripper/2.0])
        if motorstate is not None:
            self.init_data = motorstate
        self.opti.set_initial(self.var_q, self.init_data)
        # q_90_x = pin.Quaternion(pin.utils.rpyToMatrix(np.array([np.pi / 2, 0, 0])))
        # se3_90_x=pin.SE3(q_90_x,np.array([-0.1,0,0]))
        target_pose_rotation_matrix = target_pose[0:3, 0:3]
        target_pose_position = target_pose[0:3, 3]
        eef_base_matrix = pin.SE3(target_pose_rotation_matrix, target_pose_position)
        rotated_eef = eef_base_matrix #* se3_90_x
        # new_position = rotated_eef.translation
        # new_rotation_matrix = rotated_eef.rotation
        # roll, pitch, yaw = pin.rpy.matrixToRpy(new_rotation_matrix)
        # x_new, y_new, z_new = new_position[0], new_position[1], new_position[2]
        self.vis.viewer['ee_target'].set_transform(target_pose)     # for visualization

        self.opti.set_value(self.param_tf, rotated_eef.homogeneous)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            # sol = self.opti.solve()
            sol = self.opti.solve_limited()
            sol_q = self.opti.value(self.var_q)

            if self.init_data is not None:
                max_diff = max(abs(self.history_data - sol_q))
                # print("max_diff:", max_diff)
                self.init_data = sol_q
                if max_diff > 1.0/180.0*3.1415:
                    # print("Excessive changes in joint angle:", max_diff)
                    self.init_data = np.zeros(self.reduced_robot.model.nq)
                    # sol_q = (sol_q+self.history_data)/2.0
                # elif max_diff <0.5/180.0*3.1415:
                #     sol_q = self.history_data
            else:
                self.init_data = sol_q

            
            
            
            self.history_data = sol_q
            gripper_pos= np.array([0, 0])
            sol_q_extend = np.concatenate([sol_q, gripper_pos], axis=0)
            self.vis.display(sol_q_extend)  # for visualization

            if motorV is not None:
                v = motorV * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            tau_ff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v,
                              np.zeros(self.reduced_robot.model.nv))

            is_collision = self.check_self_collision(sol_q, gripper)

            return sol_q, tau_ff, not is_collision

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")
            sol_q = self.opti.debug.value(self.var_q)   # return original value
            return sol_q, '', False

    def check_self_collision(self, q, gripper=np.array([0, 0])):
        pin.forwardKinematics(self.robot.model, self.robot.data, np.concatenate([q, gripper], axis=0))
        pin.updateGeometryPlacements(self.robot.model, self.robot.data, self.geom_model, self.geometry_data)
        collision = pin.computeCollisions(self.geom_model, self.geometry_data, False)
        # print("collision:", collision)
        return collision

    def get_ik_solution(self, x,y,z,roll,pitch,yaw,gripper_pos=0,motor_state=None):
        
        q = R.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        target = pin.SE3(
            pin.Quaternion(q[3], q[0], q[1], q[2]),
            np.array([x, y, z]),
        )
        sol_q, tau_ff, get_result = self.ik_fun(target.homogeneous,0,motor_state)
        # print("result:", sol_q)
        
        if get_result :
            
            self.piper_node.joint_control_piper(sol_q[0],sol_q[1],sol_q[2],sol_q[3],sol_q[4],sol_q[5],gripper_pos)
        else :
            print("collision!!!")
            

    
class C_PiperIK():
    def __init__(self):
        self.node = rclpy.create_node('inverse_solution_node')
        # 创建Arm_IK实例
        self.arm_ik = Arm_IK()
        self.gripper_pos = 0.04
        self.pose_request = pose_request(0.15, 0, 0.2, 0, 0, 0)

        self.pose_target_old = PoseStamped()
        self.pose_target = PoseStamped()
        self.motor_state = np.zeros(6)

        
        # 启动订阅线程
        self.node.create_subscription(PoseStamped, 'piper_control/pose', self.pos_cmd_callback, 10)
        # self.node.create_subscription(Joy, 'controller/joy', self.gripper_cmd_callback, 10)
        self.node.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)

        threading.Thread(target=self.IKThread, daemon=True).start()
    
    
    def joint_state_callback(self, msg):
        # 获取关节角度
        for i in range(len(msg.name)):
            if msg.name[i] == "joint1":
                self.motor_state[0] = msg.position[i]
            elif msg.name[i] == "joint2":
                self.motor_state[1] = msg.position[i]
            elif msg.name[i] == "joint3":
                self.motor_state[2] = msg.position[i]
            elif msg.name[i] == "joint4":
                self.motor_state[3] = msg.position[i]
            elif msg.name[i] == "joint5":
                self.motor_state[4] = msg.position[i]
            elif msg.name[i] == "joint6":
                self.motor_state[5] = msg.position[i]
    
    def pos_cmd_callback(self, msg):
        # 获取Pose类型消息中的数据
        self.pose_target_old = self.pose_target
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        rpy = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]).as_euler('xyz', degrees=False)
        # 提取roll、pitch、yaw
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]
        # self.pose_request = pose_request(x, y, z, roll, pitch, yaw,time.time())
        self.pose_target = msg


    def IKThread(self):
        """Thread to continuously call the IK function"""
        rate = self.node.create_rate(200)
        time.sleep(1)
        self.node.get_logger().info("IK Thread Started")
        while rclpy.ok():
            # Call inverse kinematics with the pose from TF
            # pose_request = self.interpolate_pose_request(self.pose_request_old, self.pose_request, time.time())
            # pose_request = self.extrapolate_pose_request(self.pose_target_old, self.pose_target)
            pose_request = self.pose_target.pose
            time_start = time.time()
            roll, pitch, yaw = R.from_quat([pose_request.orientation.x, pose_request.orientation.y,
                                           pose_request.orientation.z, pose_request.orientation.w]).as_euler('xyz', degrees=False)
            self.arm_ik.get_ik_solution(pose_request.position.x,
                                        pose_request.position.y,
                                        pose_request.position.z,
                                        roll, pitch, yaw,
                                         self.gripper_pos, motor_state=self.motor_state)
            time_end = time.time()
            self.node.get_logger().info(f"IK Time: {time_end - time_start:.4f} seconds")
            rate.sleep()


def main():
    # Main function to run the node
    rclpy.init()
    piper_ik = C_PiperIK()
    rclpy.spin(piper_ik.node)
    piper_ik.node.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":

    main()


