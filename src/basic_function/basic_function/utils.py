import numpy as np
import rclpy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Header, ColorRGBA
from rclpy.duration import Duration

def Pixel2Optical(keypoints, intrinsic):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    keypoints = np.array(keypoints)
    pts = []

    for point in keypoints:
        x, y, z = float(point[0]), float(point[1]), float(point[2])  # 确保数据为浮点数

        if z == 0:  # 处理深度为 0 的情况
            pts.append([0.0, 0.0, 0.0])
            continue

        dep_x = ((x - cx) * z / fx) / 1000.0       
        dep_y = ((y - cy) * z / fy) / 1000.0         
        dep_z = z / 1000.0

        pts.append([dep_x, dep_y, dep_z])

    return np.array(pts)

# SVD calculate the plane
# calculate the fit plane
class PlaneFitter:
    def __init__(self, points):
        self.points = points
        self.plane_normal = None
        self.centroid = None

    def fit_plane(self):
     
        self.centroid = np.mean(self.points, axis=0)    
        shifted_points = self.points - self.centroid
        _, _, vh = np.linalg.svd(shifted_points)
        self.plane_normal = vh[-1]
        self.plane_normal /= np.linalg.norm(self.plane_normal)

        # detect the start point of normal vector
        start_point = 0.5 * (self.points[0] + self.points[1])
        
        return start_point, self.plane_normal

# calculate points in fixed plane
def compute_plane_vertices(normal, point, size=1.0):
    """ 根据法向量和点计算平面四个顶点 """
    A, B, C = normal

    # 计算平面内的两个方向向量，确保与法向量垂直
    # 我们选择 x 轴或者 y 轴作为起始点，根据法向量的分量来选择合适的 T1 向量
    if A != 0 or B != 0:
        T1 = np.array([-B, A, 0])  # 保证与法向量垂直
    else:
        T1 = np.array([1, 0, 0])  # 如果 A 和 B 都为零，选择 x 轴方向

    T2 = np.cross(normal, T1)  # 计算与法向量垂直的第二个方向向量

    # 归一化方向向量
    T1 = T1 / np.linalg.norm(T1) * size / 2
    T2 = T2 / np.linalg.norm(T2) * size / 2

    # 计算平面四个顶点
    P1 = point + T1 + T2
    P2 = point + T1 - T2
    P3 = point - T1 - T2
    P4 = point - T1 + T2

    return [P1, P2, P3, P4]

# 计算法向量与坐标轴夹角（单位：度）
def compute_angles(normal):
    norm = np.linalg.norm(normal)
    angle_x = np.arccos(normal[0] / norm) * 180 / np.pi
    angle_y = np.arccos(normal[1] / norm) * 180 / np.pi
    angle_z = np.arccos(normal[2] / norm) * 180 / np.pi
    return angle_x, angle_y, angle_z

def create_point_marker(points, frame_id="camera_color_optical_frame", ns="point_cloud", marker_id=0, 
                               scale=0.01, color=(0.0, 1.0, 0.0, 1.0), lifetime_sec=0):
    

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()

    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    marker.scale.x = scale
    marker.scale.y = scale

    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    marker.points = points
    marker.lifetime = Duration(sec=lifetime_sec)

    return marker

def create_plane_marker(corners, frame_id="camera_color_optical_frame") -> Marker:


    # 检查类型，如果是 list of Point，先转 numpy
    if isinstance(corners[0], Point):
        corners = np.array([[p.x, p.y, p.z] for p in corners])

    marker = Marker()
    marker.header = Header()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "plane"
    marker.id = 0
    marker.type = Marker.TRIANGLE_LIST  # 使用三角形列表绘制
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0  # 方向默认不旋转
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.pose.orientation.w = 1.0  # 无旋转

    # 颜色（RGBA）
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 0.5  # 半透明

    # 解析矩阵中的四个点
    p1 = Point(x=corners[0, 0], y=corners[0, 1], z=corners[0, 2])
    p2 = Point(x=corners[1, 0], y=corners[1, 1], z=corners[1, 2])
    p3 = Point(x=corners[2, 0], y=corners[2, 1], z=corners[2, 2])
    p4 = Point(x=corners[3, 0], y=corners[3, 1], z=corners[3, 2])

    # 组成两个三角形
    marker.points.extend([p1, p3, p2, p1, p4, p3])

    return marker

def create_arrow_marker(start_point, normal_vector, frame_id="camera_color_optical_frame", marker_id=1, scale=0.5, color=None) -> Marker:

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rclpy.clock.Clock().now().to_msg()
    marker.ns = "normal_vector"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # 设置箭头起点和终点
    start = Point(x=float(start_point[0]), y=float(start_point[1]), z=float(start_point[2]))
    end = Point(
        x=start.x + normal_vector[0] * scale,
        y=start.y + normal_vector[1] * scale,
        z=start.z + normal_vector[2] * scale
    )
    
    marker.points.append(start)
    marker.points.append(end)

    # 设置箭头的尺寸
    marker.scale.x = 0.05  # 箭杆宽度
    marker.scale.y = 0.1   # 箭头头部宽度
    marker.scale.z = 0.1   # 箭头头部高度

    # 设置颜色（默认绿色）
    if color is None:
        color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # blue
    marker.color = color

    return marker

def chest_points_vertical(
    p1, p2, p3, p4, depth_img,
    num=50,            # 在 p1–p2 连线采样的点数
    max_down=600,      # 每列向下扫描的最大像素数
    step=5,            # 扫描步长
    abs_tol=0.1,      # 绝对深度容差
    rel_tol=0.15,      # 相对深度容差
    invalid_depth=0    # 无效深度值
):
    
    valid_pts = []

    if p3[2] <100000 or p4[2]<10000:
        # not detetct hip
        h, w = depth_img.shape[:2]
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # 深度区间
        z_min, z_max = min(z1, z2), max(z1, z2)
        band_low  = z_min - abs_tol
        band_high = z_max + abs_tol
        base_scale = max(z1, z2, 1e-6)

        # 在 p1–p2 连线上采样
        xs = np.linspace(x1, x2, num)
        ys = np.linspace(y1, y2, num)
        cols = np.stack([xs, ys], axis=1)

        # 遍历每个采样点，竖直向下扫描
        for x, y in cols:
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h):
                continue

            for d in range(step, max_down+1, step):
                xj = xi
                yj = yi + d
                if not (0 <= xj < w and 0 <= yj < h):
                    break

                z = depth_img[yj, xj]
                if np.issubdtype(depth_img.dtype, np.floating) and np.isnan(z):
                    continue
                if z == invalid_depth or z <= 0:
                    continue
                z = float(z)

                # 深度条件
                in_band = (band_low <= z <= band_high)
                close_to_any = (abs(z - z1) <= rel_tol*base_scale) or \
                            (abs(z - z2) <= rel_tol*base_scale)

                if in_band or close_to_any:
                    valid_pts.append([xj, yj, z])
    else:
        # detetct hip
        valid_pts = np.vstack((p3, p4))
        

    return np.array(valid_pts, dtype=float)

def how2shootface(face_center, normal_vector, face_dist, robot_position, robot_radius):

    # Step 1: Try 0.5m forward along normal
    target_point = face_center + face_dist * normal_vector
    target_to_robot = np.linalg.norm(target_point - robot_position)

    if target_to_robot <= robot_radius:
        print('With in robot sphere')
        return target_point
    else:
        # Step 2: Ray from face_center toward robot_position, find intersection with sphere
        face_to_robot = np.linalg.norm(face_center - robot_position)
        if face_to_robot <= face_dist + robot_radius:
            print('In face sphere')
            direction_1 = target_point - robot_position
            unit_dir_1 = direction_1 / np.linalg.norm(direction_1)
            intersection_1 = robot_position + robot_radius * unit_dir_1
            return intersection_1
        else:
            print('Very far away')
            # Step 3: Find intersection with sphere
            direction_2 = face_center - robot_position
            unit_dir_2 = direction_2 / np.linalg.norm(direction_2)
            intersection_2 = robot_position + robot_radius * unit_dir_2
            return intersection_2
        
import threading
import time
import numpy as np
import bisect
import time
from collections import deque
from scipy.signal import savgol_filter
from .gesture_class import ValueSmoother

class ValueSmoother(threading.Thread):
    def __init__(self, update_callback, alpha=0.5, target_fps=60):
        super().__init__()
        self.daemon = True
        self.callback = update_callback
        self.alpha = alpha
        self.frame_interval = 1.0 / target_fps
        self.running = threading.Event()
        self.last_value = None
        self.latest_value = None
        self.latest_time = None
        self.latest_vars = None
        self.lock = threading.Lock()

    def update_value(self, new_value, timestamp, new_vars=None):
        with self.lock:
            self.latest_value = new_value
            self.latest_time = timestamp
            self.latest_vars = new_vars

    def run(self):
        try:
            next_time = time.time()
            while self.running.is_set():
                now = time.time()
                if now >= next_time:
                    self.update(now)
                    next_time += self.frame_interval
                else:
                    time.sleep(min(next_time - now, 0.001))
        except Exception as e:
            import traceback
            print(f"Exception in {self.name}: {e}")
            traceback.print_exc()

    def update(self, thread_time):
        with self.lock:
            new_value = self.latest_value
            new_timestamp = self.latest_time
            new_vars = self.latest_vars
        if new_value is not None:
            if self.last_value is None:
                self.last_value = new_value
            else:
                self.last_value = [self.alpha * new_value[i] + (1 - self.alpha) * self.last_value[i] for i in range(len(self.last_value))]
            self.callback(self.last_value, thread_time*1000, new_timestamp, new_vars)

    def start(self):
        self.running.set()
        super().start()

    def stop(self):
        self.running.clear()


class GestureAnalyzer:
    """
    统一 smoother 版本：
      - 将 pose + hand 全部关键点 flatten 合并进一个 smoother
      - 在 _on_smoothed_value 中拆开分析
    """

    def __init__(self, w, h, alpha=0.5, fps=60):
        self.camera_width = w
        self.camera_height = h

        self.swipe_time_threshold = 300   # ms
        self.hold_time_threshold = 1000   # ms

        self.reset_states()

        # hand 参数
        self.toward_threshold = 45
        self.away_threshold = 135
        self.hold_window = 1.5
        self.size_window = 2.0
        self.zero_cross_threshold = 1.0
        self.last_orientation = None
        self.last_orientation_time = None
        self.hand_size_history = deque()

        # 单一 smoother（多模态融合）
        self.smoother = ValueSmoother(
            update_callback=self._on_smoothed_value,
            alpha=alpha,
            target_fps=fps
        )
        self.smoother.start()

        self.latest_result = {
            "left_swipe": False, "right_swipe": False,
            "up_swipe": False, "down_swipe": False, "hold": False,
            "go_away": False, "come_closer": False
        }

        self._last_input_dim = 0

    # ------------------------------------------------------------
    def reset_states(self):
        self.left_start = self.right_start = self.up_start = self.down_start = self.hold_start = False
        self.left_prev_x = self.right_prev_x = 0
        self.up_start_y = self.down_start_y = 0
        self.left_start_time = self.right_start_time = 0
        self.up_start_time = self.down_start_time = 0
        self.hold_xy_history, self.hold_time_history = [], []

    # ------------------------------------------------------------
    def update(self, holistic_results):
        """
        将 pose + hand 全部关键点 flatten 合并后交给 smoother
        """
        if holistic_results is None:
            return self.latest_result

        # 提取所有存在的关键点
        all_points = []
        type_mask = []  # 记录每部分长度方便拆分

        if holistic_results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.pose_landmarks.landmark])
            all_points.extend(pose.flatten())
            type_mask.append(('pose', pose.size))

        if holistic_results.right_hand_landmarks or holistic_results.left_hand_landmarks:
            hand_lm = holistic_results.right_hand_landmarks or holistic_results.left_hand_landmarks
            handedness = 'Right' if holistic_results.right_hand_landmarks else 'Left'
            hand = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
            all_points.extend(hand.flatten())
            type_mask.append(('hand', hand.size, handedness))

        # 没有数据直接返回
        if not all_points:
            return self.latest_result

        now_ms = time.time() * 1000
        all_points = np.array(all_points)

        # 检查输入维度变化（防止 smoother 越界）
        if len(all_points) != self._last_input_dim:
            self._last_input_dim = len(all_points)
            self.smoother.last_value = all_points.copy()

        # 更新 smoother
        self.smoother.update_value(all_points, now_ms, {"mask": type_mask})

        return self.latest_result
    # ------------------------------------------------------------
    def _on_smoothed_value(self, smoothed_value, thread_time, raw_time, vars=None):
        """统一 smoother 回调：根据 mask 拆分 pose / hand"""
        if vars is None or "mask" not in vars:
            return

        lm_all = np.array(smoothed_value)
        offset = 0
        for entry in vars["mask"]:
            if entry[0] == 'pose':
                size = entry[1]
                pose_np = lm_all[offset:offset+size].reshape(-1, 3)
                offset += size
                self._analyze_pose(pose_np, thread_time)
            elif entry[0] == 'hand':
                size = entry[1]
                handedness = entry[2]
                hand_np = lm_all[offset:offset+size].reshape(-1, 3)
                offset += size
                self._analyze_hand(hand_np, handedness)

    # ------------------------------------------------------------
    def _analyze_pose(self, lm, t):
        self.latest_result.update({
            "left_swipe": self._handle_left_swipe(lm, t),
            "right_swipe": self._handle_right_swipe(lm, t),
            "up_swipe": self._handle_up_swipe(lm, t),
            "down_swipe": self._handle_down_swipe(lm, t),
            "hold": self._handle_hold(lm, t),
        })

    def _analyze_hand(self, lm, handedness):
        """基于掌心朝向与手部尺寸变化的推/拉识别（go_away / come_closer）"""
        orientation = self._get_orientation(lm, handedness)
        now = time.time()

        # 第一次进入，初始化状态
        if not hasattr(self, "_last_orientation"):
            self._last_orientation = orientation
            self._last_orientation_t = now
            self._hand_size_history = deque()
            self.latest_result.update({
                "go_away": False,
                "come_closer": False
            })
            return

        # 如果朝向变化，则重置计时和历史
        if orientation != self._last_orientation:
            self._last_orientation = orientation
            self._last_orientation_t = now
            self._hand_size_history.clear()

        # 更新手部尺寸历史
        size = self._get_hand_size(lm)
        self._update_size_history(size, now)

        # 统计零交叉次数
        zero_cross = self._count_zero_crossings()

        # 判断是否保持稳定朝向足够时间
        held = (
            self._last_orientation is not None
            and (now - self._last_orientation_t) >= self.hold_window
            and orientation == self._last_orientation
        )

        # 判定手势
        go_away = held and orientation == "PALM_TOWARD" and zero_cross >= 2
        come_closer = held and orientation == "BACK_OF_HAND" and zero_cross >= 2

        # 更新结果
        self.latest_result.update({
            "go_away": go_away,
            "come_closer": come_closer
        })

   # ------------------------------------------------------------
    # 以下为手势逻辑，与原版相同
    def _handle_left_swipe(self, lm, t):
        try:
            sl, sr, wl = lm[11][:2], lm[12][:2], lm[15][:2]
            sx = np.abs(sl[0] - sr[0])
            if sr[0] < wl[0] < sl[0]:
                if not self.left_start:
                    self.left_start = True
                self.left_start_time = t
                self.left_prev_x = wl[0]
            if not self.left_start or t - self.left_start_time > self.swipe_time_threshold:
                self.left_start = False
            if wl[0] < self.left_prev_x:
                self.left_start = False
            if wl[0] > sl[0] + sx:
                self.left_start = False
                return True
            self.left_prev_x = wl[0]
        except:
            pass
        return False

    def _handle_right_swipe(self, lm, t):
        try:
            sl, sr, wr = lm[11][:2], lm[12][:2], lm[16][:2]
            sx = np.abs(sl[0] - sr[0])
            if sr[0] < wr[0] < sl[0]:
                if not self.right_start:
                    self.right_start = True
                self.right_start_time = t
                self.right_prev_x = wr[0]
            if not self.right_start or t - self.right_start_time > self.swipe_time_threshold:
                self.right_start = False
            if wr[0] > self.right_prev_x:
                self.right_start = False
            if wr[0] < sr[0] - sx:
                self.right_start = False
                return True
            self.right_prev_x = wr[0]
        except:
            pass
        return False

    def _handle_up_swipe(self, lm, t):
        try:
            nose, sl, sr = lm[0][:2], lm[11][:2], lm[12][:2]
            sy = (sl[1] + sr[1]) / 2
            dy = np.abs(sy - nose[1])
            upper = sy + dy / 2
            lower_complete = nose[1] - dy * 1.5
            wrist = lm[16][:2]
            if nose[1] < wrist[1] < upper:
                if not self.up_start:
                    self.up_start = True
                self.up_start_time = t
                self.up_start_y = wrist[1]
            if not self.up_start or t - self.up_start_time > self.swipe_time_threshold:
                self.up_start = False
            if wrist[1] > self.up_start_y:
                self.up_start = False
            if wrist[1] < lower_complete:
                self.up_start = False
                return True
            self.up_start_y = wrist[1]
        except:
            pass
        return False

    def _handle_down_swipe(self, lm, t):
        try:
            nose, sl, sr = lm[0][:2], lm[11][:2], lm[12][:2]
            sy = (sl[1] + sr[1]) / 2
            dy = np.abs(sy - nose[1])
            upper = sy + dy / 2
            upper_complete = upper + dy * 1.5
            wrist = lm[16][:2]
            if nose[1] < wrist[1] < upper:
                if not self.down_start:
                    self.down_start = True
                self.down_start_time = t
                self.down_start_y = wrist[1]
            if not self.down_start:
                return False
            if t - self.down_start_time > self.swipe_time_threshold:
                self.down_start = False
            if wrist[1] < self.down_start_y:
                self.down_start = False
            if wrist[1] > upper_complete:
                self.down_start = False
                return True
            self.down_start_y = wrist[1]
        except:
            pass
        return False


    def _handle_hold(self, lm, t):
        try:
            wrist = lm[16][:2]
            sl, sr = lm[11][:2], lm[12][:2]
            sy = (sl[1] + sr[1]) / 2
            if wrist[1] < sy and wrist[0] < sr[0]:
                self.hold_start = True
            else:
                self.hold_start = False
                self.hold_xy_history.clear()
                self.hold_time_history.clear()
            if not self.hold_start:
                return False
            self.hold_xy_history.append(wrist)
            self.hold_time_history.append(t)
            t0 = t - self.hold_time_threshold
            if t0 < self.hold_time_history[0]:
                return False
            idx = bisect.bisect_right(self.hold_time_history, t0)
            if idx == 0:
                return False
            self.hold_xy_history = self.hold_xy_history[idx - 1:]
            self.hold_time_history = self.hold_time_history[idx - 1:]
            xs = [x[0] for x in self.hold_xy_history]
            ys = [x[1] for x in self.hold_xy_history]
            dx = (max(xs) - min(xs)) * self.camera_width
            dy = (max(ys) - min(ys)) * self.camera_height
            shoulder_dist = np.linalg.norm((sl - sr) * [self.camera_width, self.camera_height])
            if dx < shoulder_dist * 0.1 and dy < shoulder_dist * 0.1:
                return True
        except:
            pass
        return False

    # ------------------------------------------------------------
    def _get_orientation(self, hand_lm, handedness):
        """计算掌心朝向 (支持 numpy 数组输入)"""
        # hand_lm: numpy array of shape [21, 3]
        wrist = np.array([hand_lm[0, 0], hand_lm[0, 1], 1])
        middle_pip = np.array([hand_lm[10, 0], hand_lm[10, 1], 1])
        pinky_mcp = np.array([hand_lm[17, 0], hand_lm[17, 1], 1])

        palm_x = pinky_mcp - wrist
        palm_x /= np.linalg.norm(palm_x)
        palm_y = middle_pip - wrist
        palm_z = np.cross(palm_x, palm_y)
        palm_z /= np.linalg.norm(palm_z)

        if handedness == 'Left':
            palm_z = -palm_z

        camera_z = np.array([0, 0, 1])
        angle = np.degrees(np.arccos(np.clip(np.dot(palm_z, camera_z), -1, 1)))

        if angle < self.toward_threshold:
            return "PALM_TOWARD"
        elif angle > self.away_threshold:
            return "BACK_OF_HAND"
        else:
            return "SIDE"


    def _get_hand_size(self, lm):
        """
        计算手部尺寸（用于距离判断）
        lm: numpy array of shape [21, 3]
        """
        wrist = lm[0]      # [x, y, z]
        mid_mcp = lm[9]    # [x, y, z]
        size = np.sqrt((wrist[0] - mid_mcp[0])**2 + (wrist[1] - mid_mcp[1])**2)
        return float('inf') if size < 1e-6 else -1.0 / size


    def _update_size_history(self, size, now):
        """更新手部尺寸历史"""
        if not hasattr(self, "_hand_size_history"):
            self._hand_size_history = deque()
        self._hand_size_history.append((now, size))
        while self._hand_size_history and (now - self._hand_size_history[0][0]) > self.size_window:
            self._hand_size_history.popleft()

    def _count_zero_crossings(self):
        """统计手部尺寸变化曲线的过零次数"""
        if not hasattr(self, "_hand_size_history") or len(self._hand_size_history) < 3:
            return 0
        _, sizes = zip(*self._hand_size_history)
        sizes = np.array(sizes)
        smoothed = self._smooth(sizes)
        deltas = np.diff(smoothed)
        deltas[np.abs(deltas) < self.zero_cross_threshold] = 0
        signs = np.sign(deltas)
        nz = np.where(signs != 0)[0]
        if len(nz) == 0:
            return 0
        first = nz[0]
        signs[:first] = signs[first]
        for i in range(first + 1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i - 1]
        return int(np.sum(signs[1:] * signs[:-1] < 0))
    

    def _smooth(self, data, window=5):
        """
        使用 Savitzky–Golay 滤波器平滑曲线，用于检测零交叉点时减少噪声。
        data: 一维 numpy 数组
        window: 滑动窗口大小（必须为奇数）
        """
        if len(data) < window:
            return data
        return savgol_filter(data, window_length=window, polyorder=2)


    # ------------------------------------------------------------
    def close(self):
        self.smoother.stop()
