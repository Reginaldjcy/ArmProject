import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np

from .gesture_two import GestureAnalyzer   # ✅ 新的统一模型（输入为 holistic）

class GestureRecognizerNode(Node):
    def __init__(self):
        super().__init__('gesture_recognizer_node')

        # 订阅摄像头图像
        self.rgb_sub = self.create_subscription(Image, '/camera/color/image_raw', self.main_callback, 10)

        # 发布识别结果
        self.publisher = self.create_publisher(String, '/gesture_result', 10)

        # 初始化识别器（统一版本）
        self.recognizer = GestureAnalyzer(w=1280, h=720)

        self.bridge = CvBridge()

        # 初始化 Mediapipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.detector = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    # --------------------------------------------------------
    def main_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 1️⃣ Mediapipe Holistic 处理
        results_holistic = self.detector.process(img)

        # 2️⃣ 手势识别（直接返回结果字典）
        gesture_dict = self.recognizer.update(results_holistic)

        # 3️⃣ 读取识别结果（安全检查）
        if not gesture_dict:
            gesture_name = "none"
        else:
            detected = [k for k, v in gesture_dict.items() if v]
            gesture_name = detected[0] if detected else "none"

        print(gesture_dict)

        # 4️⃣ 发布 ROS 结果
        out_msg = String()
        out_msg.data = gesture_name
        self.publisher.publish(out_msg)

        # 5️⃣ 日志打印
        if gesture_name != "none":
            self.get_logger().info(f"🤖 Detected gesture: {gesture_name}")

        # 6️⃣ 可视化（仍然用 results_holistic，而不是 dict）
        self._draw_result(img, results_holistic, gesture_name)

    # --------------------------------------------------------
    def _draw_result(self, frame, results, gesture_name):
        """在图像上绘制骨架和手势文字"""
        # 绘制 pose
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # 绘制 hand
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style()
            )

        # 绘制当前手势文字
        if gesture_name != "none":
            cv2.putText(frame, f"Gesture: {gesture_name}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Gesture Recognition", frame)
        cv2.waitKey(1)

# --------------------------------------------------------
def main():
    rclpy.init()
    node = GestureRecognizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
