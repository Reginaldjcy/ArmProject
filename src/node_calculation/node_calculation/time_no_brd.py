import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from msg_interfaces.msg import TimeBool
from std_msgs.msg import Header

class BrdDistTime(Node):
    def __init__(self):
        super().__init__("brd_dist_time")  # ROS 2 节点名称

        # 创建订阅者，监听 "/node_calculation/spk_look_brd"
        self.subscription = self.create_subscription(
            Bool,
            '/node_calculation/spk_look_brd',
            self.subscription_callback,
            10  # 队列大小
        )

        # 创建发布者，发布到 "/node_calculation/time_no_brd"
        self.publisher_ = self.create_publisher(
            TimeBool,
            '/node_calculation/time_no_brd',
            10
        )

        # 配置持续时间阈值
        self.required_duration = 5  # 需要 False 持续的时间（秒）

        # 状态变量
        self.start_time = None  # 记录 False 持续开始时间
        self.is_below_threshold = False  # 记录当前是否处于 False 持续状态

    def subscription_callback(self, msg):
        """
        订阅者的回调函数，处理接收到的 Bool 消息
        """
        current_time = self.get_clock().now()  # 获取当前时间

        if not msg.data:  # 如果收到 False
            if not self.is_below_threshold:
                # 进入 False 计时状态
                self.is_below_threshold = True
                self.start_time = current_time
                self.get_logger().warn(f"Detected False at {self.start_time.to_msg()}")

            # 计算 False 持续时间
            duration = (current_time - self.start_time).nanoseconds / 1e9  # 转换为秒
            if duration >= self.required_duration:
                # False 持续足够时间，发布 True
                msg = TimeBool()

                # Header
                msg.header = Header()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'base_frame'

                # ????
                msg.result = Bool()
                msg.result.data = True

                # ??
                self.publisher_.publish(msg)
                self.get_logger().error(f"False sustained for {self.required_duration} seconds, output: True")
        else:  # 如果收到 True
            if self.is_below_threshold:
                # 复位状态
                self.is_below_threshold = False
                self.start_time = None
                self.get_logger().info("Detected True, resetting state.")

                # 立即发布 False（TimeBool 类型）
                output_msg = TimeBool()
                output_msg.header = Header()
                output_msg.header.stamp = self.get_clock().now().to_msg()
                output_msg.header.frame_id = 'base_frame'

                output_msg.result = Bool()
                output_msg.result.data = False

                self.publisher_.publish(output_msg)
                self.get_logger().info("Output: False")

def main(args=None):
    """
    主函数，初始化并运行 ROS 2 节点
    """
    rclpy.init(args=args)  # 初始化 ROS 2
    node = BrdDistTime()  # 创建节点实例
    rclpy.spin(node)  # 运行节点
    node.destroy_node()  # 退出时销毁节点
    rclpy.shutdown()  # 关闭 ROS 2

if __name__ == '__main__':
    main()
