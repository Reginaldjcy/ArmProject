import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from .utils import *

class FaceKeypoints(Node):
    def __init__(self):
        super().__init__('face_keypoints')

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
            'face_1',
            10
        )

        # CV2 bridge
        self.bridge = CvBridge()

        # Initialize Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # Whether to treat the input images as a batch of independent images.
            max_num_faces=1,          # Maximum number of faces to detect.
            min_detection_confidence=0.5,  # Minimum confidence value for face detection to be considered successful.
            min_tracking_confidence=0.5    # Minimum confidence value for the face landmarks to be considered tracked.
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
        
        # Mediapipe Face Mesh processing
        results = self.detector.process(img)

        face_1 = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # 使用 Mediapipe 的绘图工具绘制人脸关键点
                self.mp_drawing.draw_landmarks(
                    image=img,  # 需要绘制的图像
                    landmark_list=face_landmarks,  # 检测到的人脸关键点
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,  # 人脸网格连接线
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),  # 设置关键点的样式
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)  # 设置连接线的样式
                )

                # Extract keypoints as x, y coordinates (scaled to image size)
                keypoints = np.array([
                    [landmark.x * img.shape[1], landmark.y * img.shape[0]]
                    for face_landmarks in results.multi_face_landmarks
                    for landmark in face_landmarks.landmark
                ], dtype=np.float32)
        
             # Get depth
                face_1 = get_depth(keypoints, depth_image, img)

        # whether deliver data
        if face_1 is not None:
            self.face_1 = face_1
        else:
            self.face_1 = None

        # Display the image with landmarks
        cv2.imshow("Face Keypoints", img)
        cv2.waitKey(1)

        # Publish the keypoints data
        self.publisher_callback()

    def publisher_callback(self):
        if self.face_1 is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data
            self.face_1 = np.array(self.face_1, dtype=np.float32)
            float_array = Float32MultiArray()
            float_array.data = self.face_1.flatten().tolist()
            msg.matrix = float_array
            
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    face_keypoints_node = FaceKeypoints()
    rclpy.spin(face_keypoints_node)
    face_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
