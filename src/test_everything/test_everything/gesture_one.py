import numpy as np
import bisect
import time
from collections import deque
from scipy.signal import savgol_filter
from .value_smoother import ValueSmoother


class GestureAnalyzer:
    """
    使用两个独立的 smoother：
        - pose_smoother: 处理全身姿态 (33点)
        - hand_smoother: 处理手部 (21点)
    """

    def __init__(self, w, h, alpha=0.5, fps=60):
        self.camera_width = w
        self.camera_height = h

        # 阈值设置
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

        # 初始化两个独立 smoother
        self.pose_smoother = ValueSmoother(
            update_callback=self._on_pose_smoothed,
            alpha=alpha,
            target_fps=fps
        )
        self.hand_smoother = ValueSmoother(
            update_callback=self._on_hand_smoothed,
            alpha=alpha,
            target_fps=fps
        )
        self.pose_smoother.start()
        self.hand_smoother.start()

        # 最新结果
        self.latest_result = {
            "left_swipe": False, "right_swipe": False,
            "up_swipe": False, "down_swipe": False, "hold": False,
            "go_away": False, "come_closer": False
        }

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
        """输入 mediapipe holistic 结果"""
        if holistic_results is None:
            return self.latest_result

        now_ms = time.time() * 1000

        # pose
        if holistic_results.pose_landmarks:
            pose_lm = np.array([[lm.x, lm.y, lm.z] for lm in holistic_results.pose_landmarks.landmark])
            self.pose_smoother.update_value(pose_lm.flatten(), now_ms)

        # hand
        if holistic_results.right_hand_landmarks or holistic_results.left_hand_landmarks:
            hand_lm = holistic_results.right_hand_landmarks or holistic_results.left_hand_landmarks
            handedness = 'Right' if holistic_results.right_hand_landmarks else 'Left'
            hand_np = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
            self.hand_smoother.update_value(hand_np.flatten(), now_ms, {"handedness": handedness})

        setattr(holistic_results, "gesture_result", self.latest_result)
        return holistic_results

    # ------------------------------------------------------------
    def _on_pose_smoothed(self, smoothed_value, thread_time, raw_time, vars=None):
        lm = np.array(smoothed_value).reshape(-1, 3)
        self.latest_result.update({
            "left_swipe": self._handle_left_swipe(lm, thread_time),
            "right_swipe": self._handle_right_swipe(lm, thread_time),
            "up_swipe": self._handle_up_swipe(lm, thread_time),
            "down_swipe": self._handle_down_swipe(lm, thread_time),
            "hold": self._handle_hold(lm, thread_time),
        })

    def _on_hand_smoothed(self, smoothed_value, thread_time, raw_time, vars=None):
        handedness = vars["handedness"] if vars and "handedness" in vars else "Right"
        lm = np.array(smoothed_value).reshape(-1, 3)
        self.latest_result.update(self._analyze_hand(lm, handedness))

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

    def _analyze_hand(self, lm, handedness):
        class L:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z
        hand_lm = [L(*p) for p in lm]
        orientation = self._get_orientation(hand_lm, handedness)
        now = time.time()

        if orientation != self.last_orientation:
            self.last_orientation = orientation
            self.last_orientation_time = now
            self.hand_size_history.clear()

        size = self._get_hand_size(hand_lm)
        self._update_size_history(size, now)
        zero_cross = self._count_zero_crossings()
        held = (
            self.last_orientation is not None
            and (now - (self.last_orientation_time or now)) >= self.hold_window
            and orientation == self.last_orientation
        )

        go_away = held and orientation == "PALM_TOWARD" and zero_cross >= 2
        come_closer = held and orientation == "BACK_OF_HAND" and zero_cross >= 2
        return {"go_away": go_away, "come_closer": come_closer}

    def _get_orientation(self, hand_lm, handedness):
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
        wrist, mid_mcp = lm[0], lm[9]
        size = np.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)
        return float('inf') if size < 1e-6 else -1.0 / size

    def _update_size_history(self, size, now):
        self.hand_size_history.append((now, size))
        while self.hand_size_history and (now - self.hand_size_history[0][0]) > self.size_window:
            self.hand_size_history.popleft()

    def _count_zero_crossings(self):
        if len(self.hand_size_history) < 3:
            return 0
        _, sizes = zip(*self.hand_size_history)
        sizes = np.array(sizes)
        smoothed = savgol_filter(sizes, min(5, len(sizes)//2*2+1), 2)
        deltas = np.diff(smoothed)
        deltas[np.abs(deltas) < self.zero_cross_threshold] = 0
        signs = np.sign(deltas)
        nz = np.where(signs != 0)[0]
        if len(nz) == 0:
            return 0
        first = nz[0]
        signs[:first] = signs[first]
        for i in range(first+1, len(signs)):
            if signs[i] == 0:
                signs[i] = signs[i-1]
        return int(np.sum(signs[1:] * signs[:-1] < 0))

    # ------------------------------------------------------------
    def close(self):
        self.pose_smoother.stop()
        self.hand_smoother.stop()
