#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import numpy as np
from .gesture_recognizer import GestureRecognizer, HandGestureAnalyzer
from msg_interfaces.msg import TimeFloat
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

h, w, _ = 720, 1280, 3

class GestureRecognizerNode(Node):
    def __init__(self):
        super().__init__('gesture_recognizer_node')

        # 订阅姿态关键点
        self.pose_sub = self.create_subscription(TimeFloat, 'pose_1', self.pose_callback, 10)
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.main_callback, 10)

        # 发布识别结果
        self.publisher = self.create_publisher(String, '/gesture_result', 10)

        # 初始化识别器（内部包含平滑线程）
        self.recognizer = GestureRecognizer(w, h)
        self.human_1 = None
        self.bridge = CvBridge()


    def pose_callback(self, msg):
        self.human_1 = np.array(msg.matrix.data).reshape(-1,3)

    def main_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.human_1 is not None:  
            result = self.recognizer.update(self.human_1)

            # 识别出手势
            detected = [k for k, v in result.items() if v]
            gesture_name = detected[0] if detected else "none"

            # 发布结果
            out_msg = String()
            out_msg.data = gesture_name
            self.publisher.publish(out_msg)

            # 打印识别日志
            if gesture_name != "none":
                self.get_logger().info(f"🤖 Detected gesture: {gesture_name}")

def main():
    rclpy.init()
    node = GestureRecognizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()