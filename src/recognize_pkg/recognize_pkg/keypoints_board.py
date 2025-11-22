import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from msg_interfaces.msg import TimeFloat
from std_msgs.msg import Header

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from .utils import *

from torchvision import transforms as T
from .utils_board import solve_plane, solve_mask_quad, solve_depth, uv2xyz
import recognize_pkg.network as network
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import colorsys
import recognize_pkg.config as config
import torch
import cv2
import os

intrinsic = np.array([[688.4984130859375, 0.0, 639.0274047851562],
                      [0.0, 688.466552734375, 355.8525390625],
                      [0.0, 0.0, 1.0]])


class Movenet(Node):
    def __init__(self):
        super().__init__("jeff")  # Node name

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Create a publisher for Image messages
        self.publisher_ = self.create_publisher(TimeFloat, 'jeff_1', 10)
        

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1  # second
        )
        self.sync.registerCallback(self.sync_callback)

        # Publisher
        self.publisher_ = self.create_publisher(
            TimeFloat, 
            'jeff_1', 
            10)

        # CV2 bridge
        self.bridge = CvBridge()
        

        # Initialize PyTorch model and transforms
        matplotlib.use('TkAgg')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = T.Compose([
            T.ToTensor(),   
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.model = network.modeling.deeplabv3plus_mobilenet(num_classes=config.NUM_CLASSES, output_stride=config.OUTPUT_STRIDE)
        model_path = '/home/reginald/ArmProject/src/recognize_pkg/recognize_pkg/cp_wb_dl3pmn_3cls_v6.pth'
        if os.path.exists(model_path):
            self.get_logger().info(f'Loading pretrained weights from {model_path}')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        else:
            self.get_logger().info(f'no pretrained model,{model_path}')
        self.model.to(self.device)
        self.model.eval()

        # Initialize other variables
        self.matrix1 = []
        self.matrix2 = []
        self.number = 2 # choose how many contours
        self.color_mapping = np.array([colorsys.hsv_to_rgb(i/config.NUM_CLASSES, 1, 1) for i in range(config.NUM_CLASSES)])

    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""

        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')


        ############################
        ######## Detect ############
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cam_res = color_image.shape

        input_img = cv2.resize(color_image, (config.INPUT_SIZE[1], config.INPUT_SIZE[0]))
        img_tensor = self.transform(input_img).unsqueeze(0).to(self.device)
        pred = self.model(img_tensor).max(1)[1].cpu().numpy()[0]
        pred = np.array(pred)

        colorized_pred = (self.color_mapping[pred,:]*255).astype('uint8')
        cam_res = color_image.shape
        colorized_pred = cv2.resize(colorized_pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay the original image and the colorized prediction
        overlay = cv2.addWeighted(color_image, 0.5, colorized_pred, 0.5, 0)
        frame_out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Mask and solve
        pred_cam_res = cv2.resize(pred, (cam_res[1], cam_res[0]), interpolation=cv2.INTER_NEAREST)
        whiteboard_mask = (pred_cam_res == 1)
        whiteboard_full_mask = (pred_cam_res > 0)

        ############# whiteboard mask #####################
        shapes = solve_mask_quad(whiteboard_full_mask, self.number)
        if shapes is None or len(shapes) == 0:
            self.get_logger().info('No whiteboard detected')
            return
        else:
            shapes = np.array(shapes[0]).squeeze() 
            shapes = sort_pts_counterclockwise(shapes)
            
            cv2.drawContours(frame_out, [shapes], -1, (0, 0, 255), 2)
            
            # shrink
            if len(shapes) < 4:
                self.get_logger().warn(f"Expected 4 points, but got {len(shapes)}. Skipping.")
                return
            shapes = shrink_rectangle(shapes, shrink_amount=10)
            cv2.drawContours(frame_out, [shapes], -1, (0, 255, 255), 2)

            for x, y in shapes:
                if 0<y<720 and 0<x<1280:
                    continue
                else:
                    self.get_logger().warn(f"Point out of bounds: ({x}, {y})")
                    return
            self.matrix1 = get_depth(shapes, depth_image, color_image)
            
            # Display (non-blocking)
            cv2.imshow('Camera', frame_out)
            cv2.waitKey(1)  
            self.get_logger().info(f"Whiteboard detected")

            self.publisher_callback()

    def publisher_callback(self):
        """Publish the human keypoints data."""
        if self.matrix1 is not None and self.matrix2 is not None:
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_frame'

            # Data            
            float_array = Float32MultiArray()
            float_array.data = np.array(self.matrix1).flatten().tolist()

            msg.matrix = float_array
            
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    my_node = Movenet()
    try:
        rclpy.spin(my_node)
    except KeyboardInterrupt:
        pass
    finally:
        my_node.destroy_node()
        cv2.destroyAllWindows()  # Close OpenCV windows
        rclpy.shutdown()

if __name__ == '__main__':
    main()