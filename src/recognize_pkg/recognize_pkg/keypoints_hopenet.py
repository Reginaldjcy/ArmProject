import torch
import cv2
import numpy as np
import torchvision
from torchvision import transforms
from .hopenet import Hopenet 
from .utils_hopenet import *
import time
from ultralytics import YOLO  # Import YOLO from ultralytics

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
import os

def load_model(model_path, device):
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def preprocess_image(image_path, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def preprocess_frame(frame, device):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

def predict_head_pose(model, image_tensor, device):
    with torch.no_grad():
        yaw, pitch, roll = model(image_tensor)
    
    idx_tensor = torch.FloatTensor(np.arange(66)).to(device)
    softmax = torch.nn.Softmax(dim=1)

    yaw_angle = torch.sum(softmax(yaw) * idx_tensor, dim=1) * 3 - 97.5
    pitch_angle = torch.sum(softmax(pitch) * idx_tensor, dim=1) * 3 - 97.5
    roll_angle = torch.sum(softmax(roll) * idx_tensor, dim=1) * 3 - 97.5

    return yaw_angle.item(), pitch_angle.item(), roll_angle.item()

class Movenet(Node):
    def __init__(self):
        super().__init__("jeff")  # Node name

        # Subscribers for RGB and Depth images
        self.rgb_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')

        # Synchronize RGB and Depth images
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.5  # second
        )
        self.sync.registerCallback(self.sync_callback)


        # Publisher
        self.publisher_ = self.create_publisher(TimeFloat, 'hopenet_1', 10)

        # CV2 bridge
        self.bridge = CvBridge()
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Hopenet model
        self.hopenet_model = load_model("/home/reginald/ArmProject/src/recognize_pkg/recognize_pkg/hopenet_robust_alpha1.pkl", self.device)
        
        # Load YOLOv8n-face model
        self.face_model = YOLO('/home/reginald/ArmProject/src/recognize_pkg/recognize_pkg/yolov11n-face.pt')  # Make sure you have the correct model file

        # Initial
        self.pub_data = None

    def sync_callback(self, rgb_msg, depth_msg):
        """Callback for synchronized RGB and Depth images."""

        # Convert ROS Image messages to OpenCV images
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')


        # Run YOLO face detection
        results = self.face_model.predict(frame, verbose=False)
        faces = []
        face_pos = []

        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf.item()
                if conf < 0.5:  # Confidence threshold
                    continue
                
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
     
                # conv to square
                width = x2-x1
                height = y2-y1
                if width > height:
                    y1 -= int((width-height)/2)
                    y2 += int((width-height)/2)
                else:
                    x1 -= int((height-width)/2)
                    x2 += int((height-width)/2)
                
                # Ensure coordinates are within frame dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                color = (0, int(255*conf), int(255*(1-conf)))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                face = frame[y1:y2, x1:x2]
                if face.shape[0] == 0 or face.shape[1] == 0: continue
                
                faces.append(face)
                face_pos.append((x1, y1, x2, y2))
        
        # Head pose estimation for each face
        for i, face in enumerate(faces):
            try:
                image_tensor = preprocess_frame(face, self.device)
                yaw, pitch, roll = predict_head_pose(self.hopenet_model, image_tensor, self.device)
                #print(f"Yaw: {yaw:.2f}°, Pitch: {pitch:.2f}°, Roll: {roll:.2f}°")

                # Calculate center position
                tdx = (face_pos[i][0] + face_pos[i][2]) // 2
                tdy = (face_pos[i][1] + face_pos[i][3]) // 2
                draw_axis(frame, yaw, pitch, roll, tdx=tdx, tdy=tdy)
            except Exception as e:
                print(f"Error processing face: {e}")
            
            pseu_point = np.array([[tdx, tdy],
                                   [0, 0]])
            z = get_depth(pseu_point, depth_image, frame)
            z = z[0]

            self.pub_data = np.concatenate((z, np.array([roll, pitch, yaw])))                     #((z, np.array([yaw, pitch, roll])))

            # cv2.circle(frame, [int(tdx), int(tdy)], radius=30, color=(0, 255, 0), thickness=-1)
            # print(f"{[int(tdx), int(tdy)]}")
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)
            self.publisher_callback()
    
    def publisher_callback(self):
        if self.pub_data is not None: 
            msg = TimeFloat()

            # Header
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_frame"

            # Data
            float_array = Float32MultiArray()
            self.pub_data = [float(x) for x in self.pub_data]
            float_array.data = self.pub_data
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
        

