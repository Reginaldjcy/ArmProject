import rclpy
from rclpy.node import Node
from msg_interfaces.msg import TimeFloat
import pandas as pd
import time

class PoseDataRecorder(Node):
    def __init__(self):
        super().__init__('pose_data_recorder')

        # Subscriber to "pose_1"
        self.pose_subscriber = self.create_subscription(
            TimeFloat,
            'pose_1',
            self.pose_callback,
            10
        )

        # Initialize storage for pose data
        self.pose_data = []  # List to store pose data
        self.start_time = time.time()  # Track the start time
        self.next_save_time = 0  # Track the next save time (0, 0.5s, 1s, ...)

    def pose_callback(self, msg):
        """Callback to handle incoming pose_1 messages."""
        elapsed_time = time.time() - self.start_time  # Calculate relative timestamp starting from 0

        # Save data only at fixed intervals (e.g., 0, 0.5s, 1s, ...)
        if elapsed_time >= self.next_save_time:
            pose_matrix = list(msg.matrix.data)  # Convert array to list

            # Flatten the pose matrix into a single row
            row = [self.next_save_time] + pose_matrix
            self.pose_data.append(row)

            self.get_logger().info(f"Recorded data at relative timestamp: {self.next_save_time}")

            # Increment the next save time by 0.5 seconds
            self.next_save_time += 0.5

    def save_to_excel(self, file_path='/home/reginald/pose_data/pose_data.xlsx'):
        """Save collected pose data to an Excel file."""
        if self.pose_data:
            # Create a DataFrame
            column_names = ['timestamp'] + [f'value_{i}' for i in range(1, len(self.pose_data[0]))]
            df = pd.DataFrame(self.pose_data, columns=column_names)

            # Save to Excel
            df.to_excel(file_path, index=False)
            self.get_logger().info(f"Pose data saved to {file_path}")
        else:
            self.get_logger().info("No data to save.")

def main(args=None):
    rclpy.init(args=args)
    pose_data_recorder = PoseDataRecorder()

    try:
        rclpy.spin(pose_data_recorder)
    except KeyboardInterrupt:
        pose_data_recorder.save_to_excel()
    finally:
        pose_data_recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
