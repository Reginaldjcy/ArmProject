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

# --- Camera intrinsics/distortion (OpenCV rational model k1..k6, p1, p2) ---
K = np.array([[688.498,   0.000, 639.027],
              [  0.000, 688.467, 355.853],
              [  0.000,   0.000,   1.000]], dtype=np.float32)
D = np.array([47.997, -104.250, -0.000933, -0.000548, 115.033, 47.798, -103.811, 115.125], dtype=np.float32)


def load_model(model_path: str, device: torch.device) -> Hopenet:
    model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


class Movenet(Node):
    def __init__(self):
        super().__init__("jeff")

        # --- ROS I/O ---
        self.rgb_sub = Subscriber(self, Image, "/camera/color/image_raw")
        self.depth_sub = Subscriber(self, Image, "/camera/depth/image_raw")
        self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=10, slop=0.5)
        self.sync.registerCallback(self.sync_callback)

        self.publisher_ = self.create_publisher(TimeFloat, "hopenet_1", 10)
        self.bridge = CvBridge()

        # --- Compute/device ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Models ---
        self.hopenet_model = load_model(
            "/home/reginald/ArmProject/src/recognize_pkg/recognize_pkg/hopenet_robust_alpha1.pkl",
            self.device,
        )
        self.face_model = YOLO("/home/reginald/ArmProject/src/recognize_pkg/recognize_pkg/yolov11n-face.pt")

        # --- Preprocessing (reused objects to avoid re-alloc each frame) ---
        self.to_tensor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.idx_tensor = torch.arange(66, dtype=torch.float32, device=self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        # --- Camera params ---
        self.K = K
        self.D = D

        # --- Outgoing cache ---
        self.pub_data = None

    # ---- Head pose (Binned -> continuous) ----
    def predict_head_pose(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            yaw, pitch, roll = self.hopenet_model(image_tensor)
        yaw_angle = torch.sum(self.softmax(yaw) * self.idx_tensor, dim=1) * 3 - 97.5
        pitch_angle = torch.sum(self.softmax(pitch) * self.idx_tensor, dim=1) * 3 - 97.5
        roll_angle = torch.sum(self.softmax(roll) * self.idx_tensor, dim=1) * 3 - 97.5
        return yaw_angle.item(), pitch_angle.item(), roll_angle.item()

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.to_tensor(rgb).unsqueeze(0).to(self.device)
        return tensor

    def sync_callback(self, rgb_msg: Image, depth_msg: Image):
        # --- Convert ROS -> numpy ---
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # --- Undistort RGB (optional but you requested it) ---
        h, w = frame.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0.5, (w, h))
        frame = cv2.undistort(frame, self.K, self.D, None, new_K)

        # --- YOLO face detection ---
        yolo_results = self.face_model(frame, verbose=False)

        faces = []
        face_pos = []  # (x1, y1, x2, y2)
        for result in yolo_results:
            if not hasattr(result, "boxes") or result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf.item())
                if conf < 0.5:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                # make square crop
                w_box, h_box = x2 - x1, y2 - y1
                if w_box > h_box:
                    d = (w_box - h_box) // 2
                    y1 -= d
                    y2 += d
                else:
                    d = (h_box - w_box) // 2
                    x1 -= d
                    x2 += d
                # clamp to image
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w - 1, x2); y2 = min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                faces.append(frame[y1:y2, x1:x2])
                face_pos.append((x1, y1, x2, y2))

        # --- For each face: head pose + 3D point from depth ---
        last_pub = None
        for (x1, y1, x2, y2), face in zip(face_pos, faces):
            try:
                tdx = (x1 + x2) // 2
                tdy = (y1 + y2) // 2

                image_tensor = self.preprocess_frame(face)
                yaw, pitch, roll = self.predict_head_pose(image_tensor)

                # draw axis at face center (in full frame coords)
                draw_axis(frame, yaw, pitch, roll, tdx=tdx, tdy=tdy)

                # depth -> 3D (assumes get_depth returns \n                # list/array of XYZ in meters for each pixel queried)
                pixels = np.array([[tdx, tdy]], dtype=np.float32)
                xyz_list = get_depth(pixels, depth_image, frame)  # adapt to your util signature
                if xyz_list is None or len(xyz_list) == 0:
                    continue
                xyz = np.asarray(xyz_list[0]).reshape(-1)
                if xyz.size == 1:
                    # if util returns only Z, lift to XYZ in camera frame using intrinsics
                    Z = float(xyz[0])
                    X = (tdx - self.K[0, 2]) * Z / self.K[0, 0]
                    Y = (tdy - self.K[1, 2]) * Z / self.K[1, 1]
                    xyz = np.array([X, Y, Z], dtype=np.float32)

                last_pub = np.array([xyz[0], xyz[1], xyz[2], yaw, pitch, roll], dtype=np.float32)

            except Exception as e:
                self.get_logger().warn(f"Error processing face: {e}")
                continue

        # publish only the last processed face for now (or choose the largest bbox)
        if last_pub is not None:
            self.pub_data = last_pub.tolist()
            self.publisher_callback()

        # simple preview (guarded for headless)
        try:
            cv2.imshow("Camera_distort", frame)
            cv2.waitKey(1)
        except Exception:
            pass

    def publisher_callback(self):
        from msg_interfaces.msg import TimeFloat  # import here to avoid global dependency if not installed

        msg = TimeFloat()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"  # set to your camera frame

        fa = Float32MultiArray()
        fa.data = [float(x) for x in self.pub_data]
        msg.matrix = fa

        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Movenet()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
