#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import open3d as o3d

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header


class ICPMapBuilder(Node):
    def __init__(self):
        super().__init__('icp_map_builder')

        # 订阅单个点云话题
        self.subscription = self.create_subscription(
            PointCloud2, '/camera/depth_registered/points', self.cloud_callback, 10)

        # 发布融合后的点云
        self.map_pub = self.create_publisher(PointCloud2, '/icp_map', 10)

        # 保存全局地图
        self.global_map = None

        # 定时器：每 1s 处理一次
        self.timer = self.create_timer(0.01, self.process)

        # 缓存最新点云
        self.latest_cloud = None

    def cloud_callback(self, msg):
        """接收点云消息"""
        self.latest_cloud = self.ros_to_o3d(msg)
        # # self.get_logger().info("PointCloud2 fields:")
        # for f in msg.fields:
        #     self.get_logger().info(f"  name={f.name}, offset={f.offset}, datatype={f.datatype}, count={f.count}")

        # 再继续存点云
        self.latest_cloud = self.ros_to_o3d(msg)


    def process(self):
        if self.latest_cloud is None:
            return

        if self.global_map is None:
            # 第一次直接作为全局地图
            self.global_map = self.latest_cloud
            self.get_logger().info("Initialized global map with first cloud")
            return

        # ICP 配准
        threshold = 0.05
        trans_init = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.latest_cloud, self.global_map, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        self.get_logger().info(f"ICP fitness: {reg_p2p.fitness:.3f}, RMSE: {reg_p2p.inlier_rmse:.4f}")

        # 对齐当前点云
        aligned = self.latest_cloud.transform(reg_p2p.transformation)

        # 融合到全局地图
        self.global_map += aligned
        self.global_map = self.global_map.voxel_down_sample(voxel_size=0.02)  # 下采样防止爆内存

        # 发布全局点云
        ros_msg = self.o3d_to_ros(self.global_map, frame_id="camera_link")
        self.map_pub.publish(ros_msg)

    def ros_to_o3d(self, ros_pc2):
        """ROS2 PointCloud2 → Open3D PointCloud，带颜色"""
        points, colors = [], []

        for p in point_cloud2.read_points(ros_pc2, field_names=("x", "y", "z", "rgb"), skip_nans=True):
            x, y, z, rgb = p
            points.append([x, y, z])

            # unpack float32 → 4个uint8 (BGRA 顺序!)
            rgb_bytes = np.frombuffer(np.float32(rgb).tobytes(), dtype=np.uint8)
            b, g, r = rgb_bytes[0], rgb_bytes[1], rgb_bytes[2]   # 注意顺序
            colors.append([r/255.0, g/255.0, b/255.0])

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float32))
        if len(colors) > 0:
            cloud.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float32))
        return cloud

    def o3d_to_ros(self, o3d_cloud, frame_id="camera_link"):
        points = np.asarray(o3d_cloud.points)

        if points.shape[0] == 0:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = frame_id
            return point_cloud2.create_cloud_xyz32(header, [])

        if o3d_cloud.has_colors():
            colors = (np.asarray(o3d_cloud.colors) * 255).astype(np.uint8)
        else:
            colors = np.ones((points.shape[0], 3), dtype=np.uint8) * 255

        def rgb_to_float(r, g, b):
            return np.frombuffer(
                np.array([b, g, r, 255], dtype=np.uint8).tobytes(),  # BGRA !!
                dtype=np.float32
            )[0]

        rgb = np.array([rgb_to_float(r, g, b) for r, g, b in colors], dtype=np.float32)

        points_rgb = np.hstack([points.astype(np.float32), rgb.reshape(-1, 1)])

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            point_cloud2.PointField(name='x',   offset=0,  datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y',   offset=4,  datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z',   offset=8,  datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='rgb', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ]

        return point_cloud2.create_cloud(header, fields, points_rgb)




def main(args=None):
    rclpy.init(args=args)
    node = ICPMapBuilder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
