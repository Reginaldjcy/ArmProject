import numpy as np
import bisect
import time
from collections import deque
from scipy.signal import savgol_filter
from .value_smoother import ValueSmoother


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
