import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from message_filters import Subscriber, ApproximateTimeSynchronizer
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from .utils import *


class hand_keypoints(Node):
    def __init__(self):
        super().__init__('hand_keypoints')

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)
        
        # Publisher
        self.publisher_ = self.create_publisher(
            TimeFloat,
            'hand_1',
            10
        )

        # CV2 bridge
        self.bridge = CvBridge()

        # initial mediapipe hand
        self.mp_hand = mp.solutions.hands
        self.detector = self.mp_hand.Hands(
            static_image_mode=False,  # 是否静态图像模式
            max_num_hands=2,          # 最多检测几只手
            min_detection_confidence=0.5,  # 检测置信度阈值
            min_tracking_confidence=0.5              
        )
        self.mp_drawing = mp.solutions.drawing_utils


    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""
        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        ######################################################################################
        ############## mediapipe to recognize
        img = color_image.copy()
        
        # Mediapipe Hand Mesh processing
        results = self.detector.process(img)

        hand_1 = None
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                #使用 Mediapipe 的绘图工具绘制手关键点
                self.mp_drawing.draw_landmarks(
                    color_image, 
                    hand_landmarks, 
                    self.mp_hand.HAND_CONNECTIONS
                )

            # Extract keypoints as x, y coordinates (scaled to image size)
            keypoints = np.array([
                [landmark.x * img.shape[1], landmark.y * img.shape[0]] 
                for landmark in hand_landmarks.landmark
            ], dtype=np.float32)

            # Get depth
            hand_1 = get_depth(keypoints, depth_image, img)

            # 获取左右手信息
            hand_label = [1.0, 1.0, 1.0] if handedness.classification[0].label == "Left" else [0.0, 0.0, 0.0]

            # 将左右手信息附加到关键点坐标前
            hand_1 = np.concatenate(([hand_label], hand_1))

        if hand_1 is not None:
            self.hand_1 = hand_1
        else:
            self.hand_1 = None

        # Show the image    
        cv2.imshow("hand_keypoints ", color_image)
        cv2.waitKey(1)

        # Call the publisher callback to publish the datas
        self.publisher_callback()
    
    def publisher_callback(self):
        if self.hand_1 is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data
            self.hand_1 = np.array(self.hand_1, dtype=np.float32)
            float_array = Float32MultiArray()
            float_array.data = self.hand_1.flatten().tolist()
            msg.matrix = float_array

            self.publisher_.publish(msg) 


def main(args=None):
    rclpy.init(args=args)
    my_node = hand_keypoints()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
