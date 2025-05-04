import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from msg_interfaces.msg import TimeBool
from std_msgs.msg import Header
from std_msgs.msg import Float32MultiArray, Header
from msg_interfaces.msg import TimeFloat
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np

class BrdDistTime(Node):
    def __init__(self):
        super().__init__("brd_dist_time")

        # # Subscription
        # self.brd_dist = Subscriber(self, TimeFloat, '/node_calculation/brd_dist_time')
        # self.look_brd = Subscriber(self, TimeBool, '/node_calculation/time_no_brd')

        # # Synchronize two topic
        # self.sync = ApproximateTimeSynchronizer(
        #     [self.brd_dist, self.look_brd],
        #     queue_size=10,
        #     slop=0.1
        # )

        # self.sync.registerCallback(self.sync_callback)

        # 最新消息缓存
        self.last_dist_result = None
        self.last_look_result = None

        # 订阅两个话题
        self.create_subscription(TimeFloat, '/node_calculation/brd_dist_time', self.dist_callback, 10)
        self.create_subscription(TimeBool, '/node_calculation/time_no_brd', self.look_callback, 10)



        self.publisher_ = self.create_publisher(TimeFloat, 'situation_choose', 10)

    # def sync_callback(self, dist_msg, look_msg):
    #     dist_result = np.array(dist_msg.matrix.data).reshape(-1, 2)
    #     look_result = look_msg.result.data

    #     if np.allclose(dist_result, [1.0, 0.0], atol=1e-3) and look_result:
    #         # 条件一：flw_spk
    #         msg = TimeFloat()
    #         msg.header.stamp = self.get_clock().now().to_msg()
    #         msg.header.frame_id = 'base_frame'
    #         msg.matrix = Float32MultiArray(data=[11.0, 0.0])
    #         self.publisher_.publish(msg)
    #         self.get_logger().info("✅ 条件一命中：flw_spk")

    #     elif np.allclose(dist_result, [1.0, 1.0], atol=1e-3):
    #         # 条件二：act_brd_spk，不需要看 look_result
    #         msg = TimeFloat()
    #         msg.header.stamp = self.get_clock().now().to_msg()
    #         msg.header.frame_id = 'base_frame'
    #         msg.matrix = Float32MultiArray(data=[12.0, 0.0])
    #         self.publisher_.publish(msg)
    #         self.get_logger().info("✅ 条件二命中：act_brd_spk")

    #     else:
    #         # 条件三：默认
    #         self.get_logger().info(f"⚠️ 条件三命中：No Scenario | dist: {dist_result}, look: {look_result}")


    def dist_callback(self, msg):
        self.last_dist_result = np.array(msg.matrix.data).flatten()
        self.try_publish()

    def look_callback(self, msg):
        self.last_look_result = msg.result.data
        self.try_publish()
    
    def try_publish(self):
        if self.last_dist_result is None or self.last_look_result is None:
            return  # 缺一不可，等另一边更新

        dist_result = self.last_dist_result
        look_result = self.last_look_result

        if np.allclose(dist_result, [1.0, 0.0], atol=1e-3) and look_result:
            # 条件一：flw_spk
            msg = TimeFloat()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'
            msg.matrix = Float32MultiArray(data=[11.0, 0.0])
            self.publisher_.publish(msg)
            self.get_logger().info("✅ 条件一命中：flw_spk")

        elif np.allclose(dist_result, [1.0, 1.0], atol=1e-3):
            # 条件二：act_brd_spk
            msg = TimeFloat()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'
            msg.matrix = Float32MultiArray(data=[12.0, 0.0])
            self.publisher_.publish(msg)
            self.get_logger().info("✅ 条件二命中：act_brd_spk")

        else:
            self.get_logger().info(f"⚠️ 条件三命中：No Scenario | dist: {dist_result}, look: {look_result}")


def main(args=None):
    rclpy.init(args=args)
    node = BrdDistTime()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
