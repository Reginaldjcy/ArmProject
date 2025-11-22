import numpy as np
import bisect
import time
from ...basic_function.basic_function.gesture_class import ValueSmoother
from collections import deque
from scipy.signal import savgol_filter


class GestureRecognizer:
    """
    输入: MediaPipe PoseLandmarker 输出的33个关键点 (np.ndarray of shape [33, 3])
    输出：包含各手势状态的字典，例如：
          {'left_swipe': False, 'right_swipe': True, 'up_swipe': False, 'down_swipe': False, 'hold': False}
    """

    def __init__(self, w, h, alpha=0.5, fps=60):
        self.camera_width = w
        self.camera_height = h
        self.swipe_time_threshold = 300   # 毫秒
        self.hold_time_threshold = 1000   # 毫秒
        self.display_time_threshold = 500

        self.reset_states()

        # 平滑器
        self.smoother = ValueSmoother(self._on_smoothed_value, alpha=alpha, target_fps=fps)
        self.smoother.start()

        # 最新识别结果
        self.latest_result = {
            "left_swipe": False,
            "right_swipe": False,
            "up_swipe": False,
            "down_swipe": False,
            "hold": False
        }

    def reset_states(self):
        """初始化状态变量"""
        self.left_start = False
        self.left_start_time = 0
        self.left_prev_x = 0
        self.right_start = False
        self.right_start_time = 0
        self.right_prev_x = 0
        self.down_start = False
        self.down_start_time = 0
        self.down_start_y = 0
        self.up_start = False
        self.up_start_time = 0
        self.up_start_y = 0
        self.hold_start = False
        self.hold_xy_history = []
        self.hold_time_history = []

    def update(self, landmarks: np.ndarray):
        """输入 MediaPipe 的33个关键点"""
        if landmarks is None:
            return self.latest_result
        self.smoother.update_value(landmarks.flatten(), time.time() * 1000)
        return self.latest_result

    def _on_smoothed_value(self, smoothed_value, thread_time, raw_time, _):
        lm = np.array(smoothed_value).reshape(-1, 3)
        result = {
            "left_swipe": self.handle_left_swipe(lm, thread_time),
            "right_swipe": self.handle_right_swipe(lm, thread_time),
            "up_swipe": self.handle_up_swipe(lm, thread_time),
            "down_swipe": self.handle_down_swipe(lm, thread_time),
            "hold": self.handle_hold(lm, thread_time),

        }
        self.latest_result = result

    def handle_left_swipe(self, lm, t):
        try:
            sl, sr, wl = lm[11][:2], lm[12][:2], lm[15][:2]
            sx = np.abs(sl[0] - sr[0])
            if sr[0] < wl[0] < sl[0]:
                if not self.left_start:
                    self.left_start = True
                self.left_start_time = t
                self.left_prev_x = wl[0]
            if not self.left_start:
                return False
            if t - self.left_start_time > self.swipe_time_threshold:
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

    def handle_right_swipe(self, lm, t):
        try:
            sl, sr, wr = lm[11][:2], lm[12][:2], lm[16][:2]
            sx = np.abs(sl[0] - sr[0])
            if sr[0] < wr[0] < sl[0]:
                if not self.right_start:
                    self.right_start = True
                self.right_start_time = t
                self.right_prev_x = wr[0]
            if not self.right_start:
                return False
            if t - self.right_start_time > self.swipe_time_threshold:
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

    def handle_up_swipe(self, lm, t):
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
            if not self.up_start:
                return False
            if t - self.up_start_time > self.swipe_time_threshold:
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

    def handle_down_swipe(self, lm, t):
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

    def handle_hold(self, lm, t):
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
            shoulder_dist = np.linalg.norm(
                (sl - sr) * [self.camera_width, self.camera_height])
            if dx < shoulder_dist * 0.1 and dy < shoulder_dist * 0.1:
                return True
        except:
            pass
        return False

    def close(self):
        """停止平滑线程"""
        self.smoother.stop()


class HandGestureAnalyzer():
    """
    输入: Mediapipe Holistic 的结果 (results)
    输出: {'go_away': bool, 'come_closer': bool}

    特性：
      - 使用 ValueSmoother 多线程异步平滑
      - 主线程仅需调用 update(results)
      - 后台线程自动分析平滑后的手部关键点
    """

    def __init__(self,
                 toward_threshold=45,
                 away_threshold=135,
                 hold_window=1.5,
                 size_window=2.0,
                 zero_cross_threshold=1.0,
                 alpha=0.5,
                 fps=30):
        # 参数配置
        self.toward_threshold = toward_threshold
        self.away_threshold = away_threshold
        self.hold_window = hold_window
        self.size_window = size_window
        self.zero_cross_threshold = zero_cross_threshold

        # 历史状态缓存
        self.last_orientation = None
        self.last_orientation_time = None
        self.hand_size_history = deque()

        # 最新识别结果
        self.latest_result = {'go_away': False, 'come_closer': False}

        # 平滑器（后台线程）
        self.smoother = ValueSmoother(
            update_callback=self._on_smoothed_value,
            alpha=alpha,
            target_fps=fps
        )
        self.smoother.start()

    # --------------------------------------------------
    def update(self, holistic_results):
        """
        主线程接口：
        输入 Mediapipe holistic.process() 的结果。
        会自动提取手部关键点并喂入平滑器线程。
        """
        # 选择右手或左手
        if not holistic_results.right_hand_landmarks and not holistic_results.left_hand_landmarks:
            return self.latest_result

        hand_lm = holistic_results.right_hand_landmarks or holistic_results.left_hand_landmarks
        handedness = 'Right' if holistic_results.right_hand_landmarks else 'Left'

        # 展平成 numpy 数组
        flattened = []
        for lm in hand_lm.landmark:
            flattened.extend([lm.x, lm.y, lm.z])

        # 更新平滑器
        self.smoother.update_value(np.array(flattened), time.time(), {'handedness': handedness})
        return self.latest_result

    # --------------------------------------------------
    def _on_smoothed_value(self, smoothed_value, thread_time, raw_time, vars):
        """
        在后台线程中被 ValueSmoother 调用。
        """
        handedness = vars['handedness']
        lm = np.array(smoothed_value).reshape(-1, 3)
        
        hand_landmarks = self._to_named_landmarks(lm)

        # 计算掌心朝向
        orientation = self._get_orientation(hand_landmarks, handedness)
        now = time.time()

        # 状态更新
        if orientation != self.last_orientation:
            self.last_orientation = orientation
            self.last_orientation_time = now
            self.hand_size_history.clear()

        # 更新手部尺寸历史
        size = self._get_hand_size(hand_landmarks)
        self._update_size_history(size, now)

        zero_cross = self._count_zero_crossings()
        held = (
            self.last_orientation is not None
            and (now - (self.last_orientation_time or now)) >= self.hold_window
            and orientation == self.last_orientation
        )

        go_away = held and orientation == "PALM_TOWARD" and zero_cross >= 2
        come_closer = held and orientation == "BACK_OF_HAND" and zero_cross >= 2

        self.latest_result = {
            'go_away': go_away,
            'come_closer': come_closer
        }

    # --------------------------------------------------
    # ↓ 以下都是工具函数 ↓
    def _to_named_landmarks(self, lm_np):
        """把 numpy 数组转成结构方便访问"""
        class L:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        return [L(*p) for p in lm_np]

    def _get_orientation(self, hand_lm, handedness):
        """计算掌心朝向"""
        wrist = np.array([hand_lm[0].x, hand_lm[0].y, 1])
        middle_pip = np.array([hand_lm[10].x, hand_lm[10].y, 1])
        pinky_mcp = np.array([hand_lm[17].x, hand_lm[17].y, 1])

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
        """计算手部尺寸，用于距离判断"""
        wrist, mid_mcp = lm[0], lm[9]
        size = np.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)
        return float('inf') if size < 1e-6 else -1.0 / size

    def _update_size_history(self, size, now):
        """更新手部尺寸历史"""
        self.hand_size_history.append((now, size))
        while self.hand_size_history and (now - self.hand_size_history[0][0]) > self.size_window:
            self.hand_size_history.popleft()

    @staticmethod
    def _smooth(data, window=5):
        if len(data) < window:
            return data
        return savgol_filter(data, window_length=window, polyorder=2)

    def _count_zero_crossings(self):
        """统计尺寸变化曲线的过零次数"""
        if len(self.hand_size_history) < 3:
            return 0
        _, sizes = zip(*self.hand_size_history)
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

    # --------------------------------------------------
    def stop(self):
        """安全关闭 smoother 线程"""
        self.smoother.stop()
