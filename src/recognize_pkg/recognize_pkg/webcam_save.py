# save_images.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.sub = self.create_subscription(Image, '/image_raw', self.callback, 10)
        self.bridge = CvBridge()
        self.image = None
        self.counter = 0
        os.makedirs('calib_imgs', exist_ok=True)

    def callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        cv2.imshow("Camera View", self.image)
        key = cv2.waitKey(1)
        if key == ord('s'):
            filename = f'calib_imgs/img_{self.counter:03d}.png'
            cv2.imwrite(filename, self.image)
            print(f"Saved {filename}")
            self.counter += 1

def main():
    rclpy.init()
    node = ImageSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
