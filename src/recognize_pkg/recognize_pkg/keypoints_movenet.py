import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from message_filters import Subscriber, ApproximateTimeSynchronizer
from .utils import *

class Movenet(Node):
    def __init__(self):
        super().__init__("Movenet")  # Node name

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1 # second
        )
        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.publisher_ = self.create_publisher(
            TimeFloat, 
            'human_1', 
            10)

        # CV2 bridge
        self.bridge = CvBridge()

        # Load the MoveNet model
        self.model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
        self.movenet = self.model.signatures['serving_default']


    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""
        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        ##########################################################################################################
        ########### Movenet to recognize
        img = color_image.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 160, 256)
        img = tf.cast(img, dtype=tf.int32)

        # Detection section
        results = self.movenet(img)

        human_1 = None
        if results:
            keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

            # Render keypoints
            loop_through_people(color_image, keypoints_with_scores, EDGES, 0.3)

            # Scaled to image size
            height, width, c = color_image.shape
            point_2d = []
            for keypoint in keypoints_with_scores[0]:
                y, x, c = keypoint
                point_2d.append([x * width, y * height])
            
            point_2d = np.array(point_2d, dtype=np.float32)
          
            # Get depth
            human_1 = get_depth(point_2d, depth_image, color_image)
        
        # whether deliver data
        if human_1 is not None:
            self.human_1 = human_1
        else:
            self.human_1 = None

        # Show the image
        cv2.imshow("Movenet", color_image)
        cv2.waitKey(1)

        # Publish the data
        self.publisher_callback()

    def publisher_callback(self):
        """Publish the human keypoints data."""
        if self.human_1 is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data            
            self.human_1 = np.array(self.human_1, dtype=np.float32)
            float_array = Float32MultiArray()
            float_array.data = self.human_1.flatten().tolist()  # 填充数据
            msg.matrix = float_array
            
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    my_node = Movenet()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

