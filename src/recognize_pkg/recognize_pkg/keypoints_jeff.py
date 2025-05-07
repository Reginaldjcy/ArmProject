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
from .place_solver import solve_plane, solve_mask_quad, solve_depth, uv2xyz
import recognize_pkg.network as network
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import colorsys
import recognize_pkg.config as config
import torch
import cv2
import os

class Movenet(Node):
    def __init__(self):
        super().__init__("jeff")  # Node name

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

         # Create a publisher for Image messages
        self.img_publisher_ = self.create_publisher(Image, 'camera/image', 10)
        

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
        if len(shapes) >= self.number:  # 确保至少有两个矩阵
            self.matrix1 = np.squeeze(shapes[0], axis=1)  # 提取第一个矩阵
            self.matrix1 = get_depth(self.matrix1, depth_image, color_image)
            self.matrix2 = np.squeeze(shapes[1], axis=1)  # 提取第二个矩阵
            self.matrix2 = get_depth(self.matrix2, depth_image, color_image)

        for shape in shapes:
            # show the largest contours
            cv2.drawContours(frame_out, [shape], -1, (0, 0, 255), 2)

         # Convert OpenCV image to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame_out, encoding='bgr8')
        self.img_publisher_.publish(msg)

        # Display (non-blocking)
        # cv2.imshow('Camera', frame_out)
        # cv2.waitKey(1)  # Non-blocking to avoid freezing the node

        # self.get_logger().info(f'matrix 1 is {self.matrix1}')
        # self.get_logger().info(f'matrix 2 is {self.matrix2}')

        # Update human_1 (placeholder - update based on your detection logic)
        # self.publisher_callback()

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
            float_array.data = np.concatenate((self.matrix1.flatten(), self.matrix2.flatten())).tolist()
            # 使用 layout.dim 记录两个矩阵的信息
            float_array.layout.dim.append(MultiArrayDimension(label="matrix1_rows", size=self.matrix1.shape[0], stride=self.matrix1.shape[1]))
            float_array.layout.dim.append(MultiArrayDimension(label="matrix1_cols", size=self.matrix1.shape[1], stride=1))
            float_array.layout.dim.append(MultiArrayDimension(label="matrix2_rows", size=self.matrix2.shape[0], stride=self.matrix2.shape[1]))
            float_array.layout.dim.append(MultiArrayDimension(label="matrix2_cols", size=self.matrix2.shape[1], stride=1))

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