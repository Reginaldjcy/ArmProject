import rclpy
from rclpy.node import Node

from msg_interfaces.msg import TimeFloat
from message_filters import Subscriber, ApproximateTimeSynchronizer
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Bool
from msg_interfaces.msg import TimeBool

import numpy as np
from .utils import *

class flw_spk(Node):
    def __init__(self):
        super().__init__("flw_spk")  # Node name


        # Subscription
        self.human_sub = Subscriber(self, TimeFloat, 'pose_1')
        self.brd_sub = Subscriber(self, TimeFloat, 'board_1')
        self.res_sub = Subscriber(self, TimeBool, "/node_calculation/flw_spk")

        # Synchronize two topic
        self.sync = ApproximateTimeSynchronizer(
            [self.human_sub, self.brd_sub, self.res_sub],
            queue_size=10,
            slop=0.1
        )

        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.publisher_ = self.create_publisher(
             Float32MultiArray, 
             'Shooting_style', 
             10)
        
    def sync_callback(self, human_msg, board_msg, res_msg):
        human_1 = np.array(human_msg.matrix.data).reshape(-1,3)
        board_1 = np.array(board_msg.matrix.data).reshape(-1,3)
        res = res_msg.result

        # Follow speaker
        if not res:
            sum_points = human_1[0] + board_1 
            aim_points = np.mean(sum_points, axis=0)
            robot_point = point3_to_robot(aim_points)


            # Publish order
            msg = Float32MultiArray()
            msg.data = robot_point.flatten().tolist()
            self.publisher_.publish(msg) 


def main(args=None):
    rclpy.init(args=args)
    my_node = flw_spk()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
