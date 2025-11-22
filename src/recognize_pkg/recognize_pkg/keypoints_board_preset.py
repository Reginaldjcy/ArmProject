import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import numpy as np
from .utils import *

class board3d(Node):
    def __init__(self):
        super().__init__('board3d')

        # Subscription
        self.subscription = self.create_subscription(
            Image, 
            '/camera/color/image_raw',
            self.subscription_callback,             
            10)
        
        # Publisher
        self.publisher_ = self.create_publisher(
            TimeFloat,
            'board_1',
            10
        )

        # CV2 bridge
        self.bridge = CvBridge()

        # Board area
        self.board_size = [400.0, 480.0, 140.0, 240.0]
        self.board_z_depth = 1000

  
    def subscription_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        color_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # # Get the height and width of the image
        # height, width, channels = color_image.shape
    
        # # Print the height and width
        # self.get_logger().info(
        #    f"Image height: {height}, Image width: {width}")

        # # Displace board in color image
        # color_image, board_boundary = CreateBoard(color_image, self.board_size, z_depth=self.board_z_depth)

        ####################################
        ####### pre determain board ########
        # self.board_boundary = np.array([
        #     [400.0, 140.0, 1000.0],
        #     [400.0, 240.0, 1000.0],
        #     [480.0, 140.0, 1000.0],
        #     [480.0, 240.0, 1000.0]
        # ], dtype=float)
        ####################################
        ####################################

        self.board_boundary = np.array([
            [100, 10, 1000],
            [300, 10, 2500],
            [300, 710, 2500],
            [100, 710, 1000]
        ], dtype=float)
        
        # Display the image with landmarks
        for point in self.board_boundary:
            center = (int(point[0].item()), int(point[1].item()))
            # print(center)
            # print(type(center))
            cv2.circle(color_image, center, 1, (0, 0, 255), 6)
        cv2.imshow("Board 3D", color_image)
        cv2.waitKey(1)

        # Publish the keypoints data
        self.publisher_callback()

    def publisher_callback(self):
        msg = TimeFloat()

        # Header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_frame'

        # Data
        self.board_boundary = np.array(self.board_boundary, dtype=np.float32)
        
        # Initialize a Float32MultiArray and assign the data
        float_array = Float32MultiArray()
        float_array.data = self.board_boundary.flatten().tolist()  # Ensure data is correctly assigned

        # Assign the Float32MultiArray to TimeFloat's matrix field
        msg.matrix = float_array

        self.publisher_.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    face_keypoints_node = board3d()
    rclpy.spin(face_keypoints_node)
    face_keypoints_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
