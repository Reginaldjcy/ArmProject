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


class PoseKeypoints(Node):
    def __init__(self):
        super().__init__('pose_keypoints')

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
            'pose_1',
            10
        )

        # CV2 bridge
        self.bridge = CvBridge()

        # Initialize Mediapipe Pose
        self.mp_pose = mp.solutions.pose
        self.detector = self.mp_pose.Pose(
            static_image_mode=False,  # Whether to treat the input images as a batch of independent images.
            model_complexity=2,       # Complexity of the pose landmark model.
            enable_segmentation=False,  # Whether to enable segmentation mask.
            min_detection_confidence=0.3,  # Minimum confidence value for pose detection to be considered successful.
            min_tracking_confidence=0.3    # Minimum confidence value for the pose landmarks to be considered tracked.
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # FPS Calculation Variables
        self.last_frame_time = None
        self.fps = 0

    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""
        # Calculate FPS
        current_time = self.get_clock().now().seconds_nanoseconds()[0] + self.get_clock().now().seconds_nanoseconds()[1] * 1e-9
        if self.last_frame_time is not None:
            time_diff = current_time - self.last_frame_time
            self.fps = 1.0 / time_diff if time_diff > 0 else 0
        self.last_frame_time = current_time

        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Copy the image for processing
        img = color_image.copy()

        # Mediapipe Pose processing
        results = self.detector.process(img)

        pose_1 = None
        if results.pose_landmarks:
            # Draw pose landmarks on the image
            self.mp_drawing.draw_landmarks(
                color_image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )

            # Extract keypoints as x, y, z coordinates
            keypoints = np.array([
                [landmark.x * img.shape[1], landmark.y * img.shape[0]] 
                for landmark in results.pose_landmarks.landmark
            ], dtype=np.float32)

            # Get depth
            pose_1 = get_depth(keypoints, depth_image, img)

        # whether deliver data
        if pose_1 is not None:
            self.pose_1 = pose_1
        else:
            self.pose_1 = None

        # Add FPS to the image
        fps_text = f"FPS: {self.fps:.1f}"
        
        cv2.putText(
            color_image, fps_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
        )

        cv2.circle(color_image, [int(pose_1[0,0]), int(pose_1[0,1])], radius=5, color=(0, 255, 0), thickness=-1)
        # print(f"{pose_1[0,0], pose_1[0,1]}")    

        # # Display the image with landmarks
        cv2.imshow("Pose Keypoints", color_image)
        cv2.waitKey(1)

        # Publish the keypoints data
        self.publisher_callback()

    def publisher_callback(self):
        if self.pose_1 is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data
            self.pose_1 = np.array(self.pose_1, dtype=np.float32)
            float_array = Float32MultiArray()
            float_array.data = self.pose_1.flatten().tolist()
            msg.matrix = float_array

            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    pose_keypoints_node = PoseKeypoints()
    rclpy.spin(pose_keypoints_node)
    pose_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
