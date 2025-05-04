import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray 
from roarm_moveit.srv import MovePointCmd  # 导入服务类型

class MovePointCmdClient(Node):
    def __init__(self):
        super().__init__('final_pose')  # 客户端节点名称

        # create service client
        self.client = self.create_client(MovePointCmd, 'move_point_cmd')
        
        # wait service to open by service
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for the /move_point_cmd service...')
        
        self.get_logger().info('Service is available!')


        # create subscription
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'final_position',
            self.subscription_callback,
            10            
        )

    def subscription_callback(self, msg):
            x, y, z = (msg.data[0]) /5, (msg.data[1]) /5, (msg.data[2]) /5
            
            self.get_logger().info(f"Received position from topic: x={x}, y={y}, z={z}")
            self.send_request(x, y, z) 




    def send_request(self, x, y, z):
        # 创建服务请求
        request = MovePointCmd.Request()
        request.x = x
        request.y = y
        request.z = z
        
        # 打印发送请求的参数
        self.get_logger().info(f'Sending request: x={x}, y={y}, z={z}')
        
        # 异步调用服务
        self.future = self.client.call_async(request)
        return self.future


def main(args=None):
    rclpy.init(args=args)
    my_node = MovePointCmdClient()
    rclpy.spin(my_node)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

