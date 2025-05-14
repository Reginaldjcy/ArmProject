from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='webcam_node',
            parameters=[{
                'video_device': '/dev/video6',
                'image_width': 1280,
                'image_height': 720,
                'pixel_format': 'mjpeg2rgb',
                'camera_frame_id': 'usb_camera',
                'camera_info_url': 'file:///home/reginald/ArmProject/src/webcam_everything/launch/camera.yaml',

            }],
            output='screen'
        )
    ])
