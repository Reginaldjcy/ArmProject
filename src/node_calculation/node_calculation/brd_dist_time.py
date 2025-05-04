import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Header
from msg_interfaces.msg import TimeFloat
import numpy as np

class BrdDistTime(Node):
    def __init__(self):
        super().__init__("brd_dist_time")

        # ?? float32 ??
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/node_calculation/brd_edge_dist',
            self.subscription_callback,
            10
        )

        # ?? TimeFloat ????
        self.publisher_ = self.create_publisher(
            TimeFloat,
            '/node_calculation/brd_dist_time',
            10
        )

        self.threshold = 600
        self.required_duration = 5.0

        self.last_state_above = None
        self.state_start_time = None

    def subscription_callback(self, msg):
        values = list(msg.data)
        current_time = self.get_clock().now()
        is_above = any(value > self.threshold for value in values)

        # Initial
        if self.last_state_above is None:
            self.last_state_above = is_above
            self.state_start_time = current_time
            return

        if is_above == self.last_state_above:
            elapsed = (current_time - self.state_start_time).nanoseconds / 1e9
            if elapsed >= self.required_duration:
                if is_above:
                    self.publish_result([1.0, 0.0])
                    self.get_logger().warn("Longer (above threshold)")
                else:
                    self.publish_result([1.0, 1.0])
                    self.get_logger().error("Closer (below threshold)")
                self.state_start_time = current_time
        else:
            self.last_state_above = is_above
            self.state_start_time = current_time
            self.get_logger().info(f"State changed to {'Longer' if is_above else 'Closer'}.")

    def publish_result(self, array_data):
        msg = TimeFloat()

        # Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_frame'

        # ????
        float_array = Float32MultiArray()
        float_array.data = np.array(array_data, dtype=np.float32).flatten().tolist()
        msg.matrix = float_array

        # ??
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BrdDistTime()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
