import rclpy
from rclpy.node import Node
import json
import os
from std_msgs.msg import Float32MultiArray  # 修改为 Float32MultiArray

class TopicLogger(Node):
    def __init__(self, topic_name, msg_type, output_file):
        super().__init__('topic_logger')
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.output_file = output_file
        self.data = []

        # 创建文件夹路径
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # 创建订阅者
        self.subscription = self.create_subscription(
            self.msg_type,
            self.topic_name,
            self.callback,
            10
        )
        self.get_logger().info(f"Subscribed to topic: {self.topic_name}")

    def callback(self, msg):
        self.get_logger().info(f"Received message: {msg.data}")
        self.data.append(self.serialize_message(msg))

    def serialize_message(self, msg):
        # 将 Float32MultiArray 数据序列化为列表
        return list(msg.data)

    def save_data(self):
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)
        self.get_logger().info(f"Data saved to {self.output_file}")

def main(args=None):
    rclpy.init(args=args)

    # 从参数中获取设置
    topic_name = '/human_1'
    output_file = '/tmp/ros2_topic_data.json'

    # 修改为你的消息类型
    msg_type_str = 'std_msgs/Float32MultiArray'

    # 动态导入消息类型
    try:
        pkg, msg_name = msg_type_str.split('/')
        msg_module = __import__(f"{pkg}.msg", fromlist=[msg_name])
        msg_class = getattr(msg_module, msg_name)
    except (ImportError, AttributeError) as e:
        print(f"Failed to import message type: {msg_type_str}\n{e}")
        return

    # 初始化节点
    topic_logger = TopicLogger(topic_name, msg_class, output_file)

    try:
        rclpy.spin(topic_logger)
    except KeyboardInterrupt:
        topic_logger.get_logger().info("Shutting down node...")
    finally:
        topic_logger.save_data()
        topic_logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
