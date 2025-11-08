
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import time
import bisect
from enum import Enum, auto
from collections import deque
from scipy.signal import savgol_filter

# === User helper modules (provided in your workspace) ===
from value_smoother import ValueSmoother
from calc_pose_landmarks import PoseLandmarkerSolver
from video_compositor import VideoCompositor

# ================== Global Settings ==================
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
TOWARD_THRESHOLD = 45
AWAY_THRESHOLD   = 135
ZERO_CROSSING_THRESHOLD = 1
HAND_SIZE_WINDOW_SEC = 2.0
HAND_ORI_HOLD_SEC   = 1.5
RECORD_VIDEO = True
VIDEO_FPS = 60

# ================== Globals (shared state) ==================
FRAME_IMAGE = None             # latest BGR frame for drawing
FRAME_DRAWN = None             # composed frame output

# For plotting/analytics (optional video)
frames = []
timestamps = []
start_time = time.time()

# --- Hand state ---
last_hand_orientation = {'orientation': None, 'time': 0.0}
hand_size_history = deque()

# --- Pose gesture state (ONLY text & flags; drawing centralized in hand callback) ---
DISPLAY_TEXT = ''
DISPLAY_TIME = 0
DISPLAY_TIME_THRESHOLD = 500  # ms, show text this long after last trigger

# ================== Mediapipe setup ==================
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================== Utility helpers ==================
def np_to_normalized_landmark_list(lm_np):
    """Convert (21,3) array to NormalizedLandmarkList."""
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for i in range(lm_np.shape[0]):
        landmark = landmark_pb2.NormalizedLandmark()
        landmark.x = float(lm_np[i, 0])
        landmark.y = float(lm_np[i, 1])
        landmark.z = float(lm_np[i, 2])
        landmark_list.landmark.append(landmark)
    return landmark_list

class HandOrientation(Enum):
    PALM_TOWARD = auto()
    BACK_OF_HAND = auto()
    SIDE_OR_AMBIGUOUS = auto()

def getHandOrientation(hand_landmarks, handedness_str: str,
                       toward_threshold=45, away_threshold=135):
    """
    handedness_str : 'Left' or 'Right'
    Classify palm orientation relative to camera.
    """
    lm = hand_landmarks.landmark
    wrist      = np.array([lm[0].x,  lm[0].y, 1])
    middle_pip = np.array([lm[10].x, lm[10].y, 1])
    pinky_mcp  = np.array([lm[17].x, lm[17].y, 1])

    palm_x = pinky_mcp - wrist
    palm_x /= np.linalg.norm(palm_x) + 1e-9

    palm_y = middle_pip - wrist
    palm_z = np.cross(palm_x, palm_y)
    palm_z /= np.linalg.norm(palm_z) + 1e-9

    # Make normal point out of the palm consistently
    if handedness_str == 'Left':
        palm_z = -palm_z

    camera_z = np.array([0, 0, 1])
    cos_ang = np.clip(np.dot(palm_z, camera_z), -1, 1)
    angle_deg = np.degrees(np.arccos(cos_ang))

    if angle_deg < toward_threshold:
        return HandOrientation.PALM_TOWARD
    if angle_deg > away_threshold:
        return HandOrientation.BACK_OF_HAND
    return HandOrientation.SIDE_OR_AMBIGUOUS

def getHandSize(landmarks):
    wrist   = landmarks[mp_hands.HandLandmark.WRIST]
    mid_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    size = np.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)
    if size < 1e-6:
        return float('inf')
    return -1.0 / size  # larger (less negative) when closer

def smooth_savgol(data, window_size=7, polyorder=2):
    if len(data) < window_size:
        return np.array(data)
    return savgol_filter(data, window_length=window_size, polyorder=polyorder)

def updateHandSizeHistory(history, current_size, current_time, window_sec=3.0):
    history.append((current_time, current_size))
    while history and (current_time - history[0][0]) > window_sec:
        history.popleft()

def count_zero_crossings(history, threshold=1):
    if len(history) < 3:
        return 0
    _, sizes = zip(*history)
    sizes = np.array(sizes)
    smoothed = smooth_savgol(sizes, window_size=min(5, len(sizes)//2*2+1))
    deltas = np.diff(smoothed)
    deltas[np.abs(deltas) < threshold] = 0
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

def get_holistic_hands_with_handedness(frame, holistic_results, padding=0.15):
    """Return [{'label': 'Left'/'Right', 'landmarks': hand_lm}, ...] from Holistic results."""
    h, w, _ = frame.shape
    hands_info = []
    for hand_label, hand_lm in [('Left', holistic_results.left_hand_landmarks),
                                ('Right', holistic_results.right_hand_landmarks)]:
        if not hand_lm:
            continue
        hands_info.append({'label': hand_label, 'landmarks': hand_lm})
    return hands_info

# ================== Pose-based gesture logic (from v3, no drawing in callback) ==================
SWIPE_TIME_THRESHOLD = 300  # ms

LEFT_START = False
LEFT_START_TIME = 0
LEFT_PREVIOUS_X = 0

RIGHT_START = False
RIGHT_START_TIME = 0
RIGHT_PREVIOUS_X = 0

DOWN_START = False
DOWN_START_TIME = 0.0
DOWN_START_Y = 0.0

UP_START = False
UP_START_TIME = 0.0
UP_START_Y = 0.0

HOLD_START = False
HOLD_TIME_THRESHOLD = 1000  # ms
HOLD_START_XY_HISTORY = []
HOLD_START_TIME_HISTORY = []

def handle_left_swipe(lm, thread_time):
    global LEFT_START, LEFT_START_TIME, LEFT_PREVIOUS_X
    try:
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_x_dist = np.abs(shoulder_left[0] - shoulder_right[0])
        wrist_left = lm[15][:2]
        if wrist_left[0] < shoulder_left[0] and wrist_left[0] > shoulder_right[0]:
            if not LEFT_START:
                LEFT_START = True
            LEFT_START_TIME = thread_time
            LEFT_PREVIOUS_X = wrist_left[0]
        if not LEFT_START: return False
        if thread_time - LEFT_START_TIME > SWIPE_TIME_THRESHOLD:
            LEFT_START = False
        if wrist_left[0] < LEFT_PREVIOUS_X:
            LEFT_START = False
        if wrist_left[0] > shoulder_left[0] + shoulder_x_dist:
            LEFT_START = False
            return True
        LEFT_PREVIOUS_X = wrist_left[0]
    except Exception as e:
        print("Error in handle_left_swipe:", e)
    return False

def handle_right_swipe(lm, thread_time):
    global RIGHT_START, RIGHT_START_TIME, RIGHT_PREVIOUS_X
    try:
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_x_dist = np.abs(shoulder_left[0] - shoulder_right[0])
        wrist_right = lm[16][:2]
        if wrist_right[0] < shoulder_left[0] and wrist_right[0] > shoulder_right[0]:
            if not RIGHT_START:
                RIGHT_START = True
            RIGHT_START_TIME = thread_time
            RIGHT_PREVIOUS_X = wrist_right[0]
        if not RIGHT_START: return False
        if thread_time - RIGHT_START_TIME > SWIPE_TIME_THRESHOLD:
            RIGHT_START = False
        if wrist_right[0] > RIGHT_PREVIOUS_X:
            RIGHT_START = False
        if wrist_right[0] < shoulder_right[0] - shoulder_x_dist:
            RIGHT_START = False
            return True
        RIGHT_PREVIOUS_X = wrist_right[0]
    except Exception as e:
        print("Error in handle_right_swipe:", e)
    return False

def handle_down_swipe(lm, thread_time):
    global DOWN_START, DOWN_START_TIME, DOWN_START_Y
    try:
        nose = lm[0][:2]
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
        bound_y_dist = np.abs(shoulder_y - nose[1])
        upper_bound = shoulder_y + bound_y_dist/2
        upper_bound_complete = upper_bound + bound_y_dist/2*3
        wrist = lm[16][:2]
        if nose[1] < wrist[1] < upper_bound:
            if not DOWN_START:
                DOWN_START = True
            DOWN_START_TIME = thread_time
            DOWN_START_Y    = wrist[1]
        if not DOWN_START: return False
        if thread_time - DOWN_START_TIME > SWIPE_TIME_THRESHOLD:
            DOWN_START = False; return False
        if wrist[1] < DOWN_START_Y:
            DOWN_START = False; return False
        if wrist[1] > upper_bound_complete:
            DOWN_START = False; return True
        DOWN_START_Y = wrist[1]
    except Exception as e:
        print("Error in handle_down_swipe:", e)
    return False

def handle_up_swipe(lm, thread_time):
    global UP_START, UP_START_TIME, UP_START_Y
    try:
        nose = lm[0][:2]
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
        bound_y_dist = np.abs(shoulder_y - nose[1])
        upper_bound = shoulder_y + bound_y_dist/2
        lower_bound_complete = nose[1] - bound_y_dist/2*3
        wrist = lm[16][:2]
        if nose[1] < wrist[1] < upper_bound:
            if not UP_START:
                UP_START = True
            UP_START_TIME = thread_time
            UP_START_Y    = wrist[1]
        if not UP_START: return False
        if thread_time - UP_START_TIME > SWIPE_TIME_THRESHOLD:
            UP_START = False; return False
        if wrist[1] > UP_START_Y:
            UP_START = False; return False
        if wrist[1] < lower_bound_complete:
            UP_START = False; return True
        UP_START_Y = wrist[1]
    except Exception as e:
        print("Error in handle_up_swipe:", e)
    return False

def handle_hold(lm, thread_time):
    global HOLD_START, HOLD_START_XY_HISTORY, HOLD_START_TIME_HISTORY
    try:
        wrist = lm[16][:2]
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1] + shoulder_right[1]) / 2
        if wrist[1] < shoulder_y and wrist[0] < shoulder_right[0]:
            HOLD_START = True
        else:
            HOLD_START = False
            HOLD_START_XY_HISTORY = []
            HOLD_START_TIME_HISTORY = []
        if not HOLD_START: return False
        HOLD_START_XY_HISTORY.append(wrist)
        HOLD_START_TIME_HISTORY.append(thread_time)
        hold_start_time = thread_time - HOLD_TIME_THRESHOLD
        if len(HOLD_START_TIME_HISTORY) == 0 or hold_start_time < HOLD_START_TIME_HISTORY[0]:
            return False
        window_start_idx = bisect.bisect_right(HOLD_START_TIME_HISTORY, hold_start_time)
        if window_start_idx == 0:
            return False
        HOLD_START_XY_HISTORY = HOLD_START_XY_HISTORY[window_start_idx-1:]
        HOLD_START_TIME_HISTORY = HOLD_START_TIME_HISTORY[window_start_idx-1:]
        window_max_x = max(HOLD_START_XY_HISTORY, key=lambda x: x[0])[0]
        window_min_x = min(HOLD_START_XY_HISTORY, key=lambda x: x[0])[0]
        window_max_y = max(HOLD_START_XY_HISTORY, key=lambda x: x[1])[1]
        window_min_y = min(HOLD_START_XY_HISTORY, key=lambda x: x[1])[1]
        # Shoulder distance as normalization
        dx = (shoulder_left[0] - shoulder_right[0]) * CAMERA_WIDTH
        dy = (shoulder_left[1] - shoulder_right[1]) * CAMERA_HEIGHT
        shoulder_distance_px = np.sqrt(dx**2 + dy**2)
        window_size_x = (window_max_x - window_min_x) * CAMERA_WIDTH
        window_size_y = (window_max_y - window_min_y) * CAMERA_HEIGHT
        window_threshold = shoulder_distance_px * 0.1
        if window_size_x < window_threshold and window_size_y < window_threshold:
            return True
    except Exception as e:
        print("Error in handle_hold:", e)
    return False

# === Pose smoother callback: only update gesture text, avoid drawing here ===
def pose_smoother_callback(values, thread_time, raw_timestamp, og_vars):
    global DISPLAY_TEXT, DISPLAY_TIME
    try:
        lm = np.array(values).reshape(-1, 3)
        pose_landmarks = lm[0:33]  # ignore world landmarks for gesture logic
        is_left_swipe  = handle_left_swipe(pose_landmarks, thread_time)
        is_right_swipe = handle_right_swipe(pose_landmarks, thread_time)
        is_down_swipe  = handle_down_swipe(pose_landmarks, thread_time)
        is_up_swipe    = handle_up_swipe(pose_landmarks, thread_time)
        is_hold        = handle_hold(pose_landmarks, thread_time)

        if is_left_swipe:
            DISPLAY_TEXT = 'left swipe';  DISPLAY_TIME = thread_time
        elif is_right_swipe:
            DISPLAY_TEXT = 'right swipe'; DISPLAY_TIME = thread_time
        elif is_down_swipe:
            DISPLAY_TEXT = 'down swipe';  DISPLAY_TIME = thread_time
        elif is_up_swipe:
            DISPLAY_TEXT = 'up swipe';    DISPLAY_TIME = thread_time
        elif is_hold:
            DISPLAY_TEXT = 'hold';        DISPLAY_TIME = thread_time
        else:
            # clear if expired
            if thread_time - DISPLAY_TIME > DISPLAY_TIME_THRESHOLD:
                DISPLAY_TEXT = ''
    except Exception as e:
        print("Error in pose_smoother_callback:", e)

def pose_landmarker_callback(smoother, result, output_image, timestamp_ms):
    try:
        if result.pose_landmarks and result.pose_world_landmarks:
            flat_lm = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]]).reshape(-1)
            flat_world_lm = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]]).reshape(-1)
            smoother.update_value(np.concatenate((flat_lm, flat_world_lm)), timestamp_ms)
    except Exception as e:
        print("Error in pose_landmarker_callback:", e)

# ================== Hand smoother callback: central drawing & composition ==================
def hand_smoothed_callback(smoothed_value, thread_time_ms, original_time, og_vars):
    global FRAME_IMAGE, FRAME_DRAWN
    global ZERO_CROSSING_THRESHOLD, last_hand_orientation, hand_size_history
    global DISPLAY_TEXT, DISPLAY_TIME

    if FRAME_IMAGE is None:
        return
    image = FRAME_IMAGE.copy()

    lm = np.array(smoothed_value).reshape(-1, 3)
    hand_handedness = og_vars.get('label', 'Right')
    hand_landmarks = np_to_normalized_landmark_list(lm)

    # Draw hand landmarks
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Orientation & size
    hand_orientation = getHandOrientation(hand_landmarks, hand_handedness,
                                          toward_threshold=TOWARD_THRESHOLD,
                                          away_threshold=AWAY_THRESHOLD)

    if hand_orientation != last_hand_orientation['orientation']:
        last_hand_orientation['orientation'] = hand_orientation
        last_hand_orientation['time'] = time.time()
        hand_size_history.clear()

    if hand_orientation == HandOrientation.PALM_TOWARD:
        orientation = "Palm Toward Camera"
    elif hand_orientation == HandOrientation.BACK_OF_HAND:
        orientation = "Back of Hand Toward Camera"
    else:
        orientation = "Side / Ambiguous"

    cv2.putText(image, orientation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    size = getHandSize(hand_landmarks.landmark)
    updateHandSizeHistory(hand_size_history, size, time.time(), window_sec=HAND_SIZE_WINDOW_SEC)
    cv2.putText(image, f"Hand size: {size:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    zero_crossings = count_zero_crossings(hand_size_history, threshold=ZERO_CROSSING_THRESHOLD)
    cv2.putText(image, f"Zero Crossings: {zero_crossings}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(image, f"Last Hand Orientation: {last_hand_orientation['orientation']}, Time: {last_hand_orientation['time']:.2f}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Gesture based on hand + oscillations
    is_hand_orientation_held = (last_hand_orientation['orientation'] == hand_orientation) and \
                               (time.time() - last_hand_orientation['time'] >= HAND_ORI_HOLD_SEC)
    action = 'Nothing'
    if is_hand_orientation_held and hand_orientation == HandOrientation.PALM_TOWARD and zero_crossings >= 2:
        action = 'Go Away'
    elif is_hand_orientation_held and hand_orientation == HandOrientation.BACK_OF_HAND and zero_crossings >= 2:
        action = 'Come Closer'
    cv2.putText(image, f"Action: {action}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Held: {is_hand_orientation_held}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # === Overlay latest pose gesture text (from pose smoother) ===
    if DISPLAY_TEXT:
        cv2.putText(image, f"Pose: {DISPLAY_TEXT}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)

    FRAME_DRAWN = image

# ================== Main ==================
def main():
    global FRAME_IMAGE, FRAME_DRAWN, frames, timestamps

    # Start smoothers
    hand_smoother = ValueSmoother(update_callback=hand_smoothed_callback, alpha=0.5, target_fps=30)
    hand_smoother.start()

    pose_landmark_smoother = ValueSmoother(pose_smoother_callback, alpha=0.2, target_fps=60)
    pose_landmark_smoother.start()

    # Pose Landmarker (async)
    pose_landmark_solver = PoseLandmarkerSolver(
        lambda res, img, t: pose_landmarker_callback(pose_landmark_smoother, res, img, t),
        "pose_landmarker_full.task"
    )

    print("Starting Webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    videoCompositor = VideoCompositor(VIDEO_FPS, (CAMERA_WIDTH, CAMERA_HEIGHT)) if RECORD_VIDEO else None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Error reading frame')
                break

            # Update global source frame
            FRAME_IMAGE = frame.copy()

            # 1) Pose async
            pose_landmark_solver.solve_async(frame)

            # 2) Hands sync via Holistic
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)

            # Feed first detected hand landmarks to smoother
            hands_info = []
            if results:
                hands_info = get_holistic_hands_with_handedness(image_rgb, results)
            if len(hands_info) > 0 and hands_info[0]['label'] is not None:
                flattened_lms = []
                for lm in hands_info[0]['landmarks'].landmark:
                    flattened_lms.extend([lm.x, lm.y, lm.z])
                hand_smoother.update_value(np.array(flattened_lms), int(time.time() * 1000),
                                           new_vars={'label': hands_info[0]['label']})

            # 3) Show composed frame from hand callback
            if FRAME_DRAWN is not None:
                cv2.imshow("Hand & Pose Control", FRAME_DRAWN)
                if RECORD_VIDEO and videoCompositor is not None:
                    videoCompositor.add_frame(FRAME_DRAWN, int(time.time() * 1000))
                frames.append(FRAME_DRAWN)
                timestamps.append(time.time() - start_time)
                FRAME_DRAWN = None  # consumed

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Quitting...')
                break

    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()
        pose_landmark_solver.close()
        hand_smoother.stop()
        pose_landmark_smoother.stop()
        hand_smoother.join()
        pose_landmark_smoother.join()
        cap.release()

        if RECORD_VIDEO and videoCompositor is not None:
            videoCompositor.save_video('output.mp4')

        # Also save a simple constant-FPS MP4 of displayed frames (optional)
        if len(frames) > 0:
            print("Saving fallback video...")
            fps = VIDEO_FPS
            video = cv2.VideoWriter('output_fallback.mp4',
                                    cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps,
                                    (frames[0].shape[1], frames[0].shape[0]))
            for f in frames:
                video.write(f)
            video.release()

if __name__ == "__main__":
    main()
