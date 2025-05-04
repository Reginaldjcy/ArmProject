import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from msg_interfaces.msg import TimeFloat
from message_filters import Subscriber, ApproximateTimeSynchronizer

from cv_bridge import CvBridge
import cv2
import numpy as np
from .utils import *

class brd_edge_dist(Node):
    def __init__(self):
        super().__init__("brd_edge_dist") # node name

        # Subscription
        self.human_sub = Subscriber(self, TimeFloat, 'pose_1')
        self.brd_sub = Subscriber(self, TimeFloat, 'board_1')
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')

        # Synchronize two topic
        self.sync = ApproximateTimeSynchronizer(
            [self.human_sub, self.brd_sub, self.rgb_sub],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.publisher_ = self.create_publisher(
             Float32MultiArray, 
             '/node_calculation/brd_edge_dist', 
             10)
        
        # CV2 bridge
        self.bridge = CvBridge()
        
    def sync_callback(self, human_msg, board_msg, rgb_msg):
        human_1 = np.array(human_msg.matrix.data).reshape(-1,3)
        board_boundary = np.array(board_msg.matrix.data).reshape(-1,3)
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')

        # show the result of left and right
        color_image, relative_position = speaker_board_left_right(human_1, board_boundary, color_image)

        # Determine relative distance
        relative_dist, color_image = speaker_board_dist(color_image, relative_position, human_1, board_boundary, percentile = 5, coef = 1.15e3 )
        
        # result out for next topic
        self.relative_dist = relative_dist
        print(relative_dist)

        cv2.imshow("brd_edge_dist", color_image)
        cv2.waitKey(1)

        # Call the publisher callback to publish the data
        self.publisher_callback()
    
    def publisher_callback(self):
        msg = Float32MultiArray()
        msg.data = self.relative_dist.flatten().tolist()
        self.publisher_.publish(msg) 

def main(args=None):
    rclpy.init(args=args)
    my_node = brd_edge_dist()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
