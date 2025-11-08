import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import time
from enum import Enum, auto
from collections import deque
from scipy.signal import savgol_filter
from value_smoother import ValueSmoother

ZERO_CROSSING_THRESHOLD = 1

FRAME_IMAGE = None
FRAME_DRAWN = None

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,  # more accurate around lips and eyes
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def np_to_normalized_landmark_list(lm_np):
    """
    lm_np: shape (21, 3), x, y, z normalized coordinates
    Returns: NormalizedLandmarkList
    """
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for i in range(lm_np.shape[0]):
        landmark = landmark_pb2.NormalizedLandmark()
        landmark.x = float(lm_np[i, 0])
        landmark.y = float(lm_np[i, 1])
        landmark.z = float(lm_np[i, 2])
        landmark_list.landmark.append(landmark)
    return landmark_list

def get_holistic_hands_with_handedness(frame, holistic_results, padding=0.15):
    h, w, _ = frame.shape
    hands_info = []

    for hand_label, hand_lm in [('Left' , holistic_results.left_hand_landmarks), 
                           ('Right', holistic_results.right_hand_landmarks)]:
        if not hand_lm:
            continue

        # Compute hand bounding box from landmarks
        xs = [lm.x for lm in hand_lm.landmark]
        ys = [lm.y for lm in hand_lm.landmark]
        xmin = max(int((min(xs) - padding) * w), 0)
        xmax = min(int((max(xs) + padding) * w), w)
        ymin = max(int((min(ys) - padding) * h), 0)
        ymax = min(int((max(ys) + padding) * h), h)

        if xmax - xmin < 5 or ymax - ymin < 5:
            continue  # skip tiny boxes

        # Crop and convert to RGB
        crop = cv2.cvtColor(frame[ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)

        # Run Hands detector on crop
        # res = hands.process(crop)
        label = hand_label
        out_hand_lm = hand_lm
        # if res.multi_hand_landmarks and res.multi_handedness:
        #     # Pick the first detection (usually only one in crop)
        #     label = res.multi_handedness[0].classification[0].label
        #     out_hand_lm = res.multi_hand_landmarks[0]

        hands_info.append({'label': label, 'landmarks': hand_lm})

    return hands_info

class HandOrientation(Enum):
    PALM_TOWARD = auto()
    BACK_OF_HAND = auto()
    SIDE_OR_AMBIGUOUS = auto()

def getHandOrientation(hand_landmarks, handedness_str: str,
                       toward_threshold=45, away_threshold=135):
    """
    handedness_str : 'Left' or 'Right'  (exactly what Mediapipe gives)
    Returns one of the three HandOrientation enums.
    """
    # 1. Build a little 3-D palm coordinate system
    lm = hand_landmarks.landmark
    # wrist      = np.array([lm[0].x,  lm[0].y,  lm[0].z])
    # middle_pip = np.array([lm[10].x, lm[10].y, lm[10].z])
    # pinky_mcp = np.array([lm[17].x, lm[17].y, lm[17].z])
    wrist      = np.array([lm[0].x,  lm[0].y, 1])
    middle_pip = np.array([lm[10].x, lm[10].y, 1])
    pinky_mcp = np.array([lm[17].x, lm[17].y, 1])

    # x-axis : along the palm toward the little finger
    palm_x = pinky_mcp - wrist
    palm_x /= np.linalg.norm(palm_x)

    # z-axis : “out of the palm” (cross product gives a vector perpendicular to palm plane)
    palm_y = middle_pip - wrist          # proximal direction of middle finger
    palm_z = np.cross(palm_x, palm_y)    # points outward from palm
    palm_z /= np.linalg.norm(palm_z)

    # 2. Make sure normal always points **out** of the palm for both left and right hands
    #    (for a right hand the anatomical normal is inverted w.r.t. left)

    if handedness_str == 'Left':
        palm_z = -palm_z

    # 3. Camera looks along **positive z** in the image coordinate system (x→right, y→down, z→forward)
    camera_z = np.array([0, 0, 1])

    # 4. Angle between palm normal and camera
    cos_ang = np.clip(np.dot(palm_z, camera_z), -1, 1)

    angle_deg = np.degrees(np.arccos(cos_ang))

    # 5. Classify
    if angle_deg < toward_threshold:
        return HandOrientation.PALM_TOWARD
    if angle_deg > away_threshold:
        return HandOrientation.BACK_OF_HAND
    return HandOrientation.SIDE_OR_AMBIGUOUS

def getHandSize(landmarks):
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    mid_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    size = np.sqrt((wrist.x - mid_mcp.x)**2 + (wrist.y - mid_mcp.y)**2)

    if size < 1e-6:
        return float('inf')
    return -1.0 / size

def smooth_savgol(data, window_size=7, polyorder=2):
    if len(data) < window_size:
        return data
    return savgol_filter(data, window_length=window_size, polyorder=polyorder)

# video variables
frames = []
timestamps = []
start_time = time.time()

# calc var
camera_direction = np.array([0, 0, -1])
TOWARD_THRESHOLD = 45
AWAY_THRESHOLD = 135

last_hand_orientation = {'orientation':None, 'time':None}
hand_orientation_hold_window = 1.5
hand_size_queue_window = 2
hand_size_history = deque()

def updateHandSizeHistory(hand_size_history, current_size, current_time, window_sec=3.0):
    hand_size_history.append((current_time, current_size))
    
    while hand_size_history and (current_time - hand_size_history[0][0]) > window_sec:
        hand_size_history.popleft()

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
line1, = ax1.plot([], [], label='Hand Size')
ax1.set_ylabel('Size')
ax1.set_title('Hand Size Over Time (Last 3 sec)')
ax1.set_ylim(-20, 0)
ax1.legend()

line2, = ax2.plot([], [], label='Delta Hand Size')
ax2.set_ylabel('Delta Size')
ax2.set_xlabel('Time (s)')
ax2.set_title('Delta Hand Size Over Time (Last 3 sec)')
ax2.set_ylim(-5, 5)
ax2.legend()

def update_plot(hand_size_history, window_sec=3.0, show=True):
    global fig, line1, line2, ax1, ax2
    if len(hand_size_history) < 5:
        return None

    times, sizes = zip(*hand_size_history)
    times = np.array(times)
    sizes = np.array(sizes)
    # times_shifted = times - times[-1] + window_sec

    smoothed_sizes = smooth_savgol(sizes, window_size=5)
    smoothed_times = times[-len(smoothed_sizes):]
    smoothed_times_shifted = smoothed_times - smoothed_times[-1] + window_sec

    deltas = np.diff(smoothed_sizes)
    delta_times = smoothed_times_shifted[1:]

    line1.set_data(smoothed_times_shifted, smoothed_sizes)
    line2.set_data(delta_times, deltas)

    # Only set limits once, or if necessary
    ax1.set_xlim(0, window_sec)
    ax2.set_xlim(0, window_sec)

    if show:
        # update the interactive figure window
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        return fig
    else:
        # render to numpy array instead
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)
        return img

def overlay_numpy_fig(frame, npfig, size=(320, 240), alpha=1.0, margin=10):
    # Convert RGB to BGR
    plot_bgr = cv2.cvtColor(npfig, cv2.COLOR_RGB2BGR)

    # Resize
    plot_bgr = cv2.resize(plot_bgr, size)
    plot_h, plot_w = plot_bgr.shape[:2]

    # Compute coordinates
    frame_h, frame_w = frame.shape[:2]
    x1 = max(frame_w - plot_w - margin, 0)
    y1 = margin
    x2 = x1 + plot_w
    y2 = y1 + plot_h

    # Ensure slice is valid
    frame_slice = frame[y1:y2, x1:x2]
    # If frame_slice is smaller than plot_bgr, resize plot_bgr to match
    if frame_slice.shape[:2] != plot_bgr.shape[:2]:
        plot_bgr = cv2.resize(plot_bgr, (frame_slice.shape[1], frame_slice.shape[0]))

    # Blend or replace
    if alpha >= 1.0:
        frame[y1:y2, x1:x2] = plot_bgr
    else:
        frame[y1:y2, x1:x2] = cv2.addWeighted(plot_bgr, alpha, frame[y1:y2, x1:x2], 1-alpha, 0)

    return frame

def count_zero_crossings(hand_size_history, threshold=1):
    """
    Count sign changes in the derivative of smoothed hand size,
    ignoring changes smaller than `threshold`.
    hand_size_history: list of (time, size) tuples, already pruned to last window.
    threshold: minimal delta magnitude to consider (in same units as size).
    """
    if len(hand_size_history) < 3:
        return 0

    _, sizes = zip(*hand_size_history)
    sizes = np.array(sizes)

    # Smooth first (e.g., Savitzky-Golay or moving average)
    smoothed = smooth_savgol(sizes, window_size=5)  # ensure window_size is odd and <= len(sizes)
    
    # Compute deltas
    deltas = np.diff(smoothed)

    # Zero out small changes
    deltas[np.abs(deltas) < threshold] = 0

    # Compute sign array, but we want to ignore zero segments when detecting crossings.
    signs = np.sign(deltas)  # gives -1, 0, +1

    # Forward-fill zeros so they don't spuriously break runs:
    # If a zero occurs, copy the previous non-zero sign; if leading zeros, drop them.
    # First find first non-zero sign
    nonzero_indices = np.where(signs != 0)[0]
    if len(nonzero_indices) == 0:
        return 0
    first_nz = nonzero_indices[0]
    # For leading zeros, set to first non-zero sign
    signs[:first_nz] = signs[first_nz]
    # Forward fill subsequent zeros
    for i in range(first_nz+1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]

    # Now count sign changes between consecutive deltas
    zero_crossings = np.sum(signs[1:] * signs[:-1] < 0)
    return int(zero_crossings)

# --------------------

def smoothed_callback(smoothed_value, thread_time_ms, original_time, og_vars):
    global FRAME_IMAGE, FRAME_DRAWN
    global ZERO_CROSSING_THRESHOLD
    if FRAME_IMAGE is None:
        return
    FRAME_DRAWN = FRAME_IMAGE.copy()

    lm = np.array(smoothed_value).reshape(-1, 3)

    hand_handedness = og_vars['label']
    hand_landmarks = np_to_normalized_landmark_list(lm)

    mp_drawing.draw_landmarks(FRAME_DRAWN, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    hand_orientation = getHandOrientation(hand_landmarks, hand_handedness, toward_threshold=TOWARD_THRESHOLD, away_threshold=AWAY_THRESHOLD)
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
    updateHandSizeHistory(hand_size_history, size, time.time(), window_sec=hand_size_queue_window)
    cv2.putText(image, f"Hand size: {size:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    # np_fig = update_plot(hand_size_history, window_sec=hand_size_queue_window, show=False)
    # print(np_fig is not None)
    # if np_fig is not None:
    #     overlay_numpy_fig(image, np_fig, alpha=0.5)

    zero_crossings = count_zero_crossings(hand_size_history, threshold=ZERO_CROSSING_THRESHOLD)
    cv2.putText(image, f"Zero Crossings: {zero_crossings}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(image, f"Last Hand Orientation: {last_hand_orientation['orientation']}, Time: {last_hand_orientation['time']:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # test for gesture conditions
    is_hand_orientation_held = (last_hand_orientation['orientation']==hand_orientation) and (time.time()-last_hand_orientation['time'] >= hand_orientation_hold_window)
    action = 'Nothing'
    if is_hand_orientation_held and hand_orientation == HandOrientation.PALM_TOWARD and zero_crossings >= 2:
        print("go away action")
        action = 'Go Away'
    elif is_hand_orientation_held and hand_orientation == HandOrientation.BACK_OF_HAND and zero_crossings >= 2:
        print("come closer action")
        action = 'Come Closer'
    cv2.putText(image, f"Action: {action}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(image, f"Held: {is_hand_orientation_held}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


hand_smoother = ValueSmoother(update_callback=smoothed_callback, alpha=0.5, target_fps=30)
hand_smoother.start()

# --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue        
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(image_rgb)

    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # --------------------

    hands_info = get_holistic_hands_with_handedness(image_rgb, results)
    FRAME_IMAGE = image

    if len(hands_info) > 0 and hands_info[0]['label'] is not None:
        flattened_lms = []
        for lm in hands_info[0]['landmarks'].landmark:
            flattened_lms.extend([lm.x, lm.y, lm.z])
        hand_smoother.update_value(np.array(flattened_lms), time.time(), new_vars={'label': hands_info[0]['label']})

    # for hand_info in hands_info:
    #     hand_landmarks = hand_info['landmarks']
    #     hand_handedness = hand_info['label']

    #     mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
    #     hand_orientation = getHandOrientation(hand_landmarks, hand_handedness, toward_threshold=TOWARD_THRESHOLD, away_threshold=AWAY_THRESHOLD)
    #     if hand_orientation != last_hand_orientation['orientation']:
    #         last_hand_orientation['orientation'] = hand_orientation
    #         last_hand_orientation['time'] = time.time()
        
    #     if hand_orientation == HandOrientation.PALM_TOWARD:
    #         orientation = "Palm Toward Camera"
    #     elif hand_orientation == HandOrientation.BACK_OF_HAND:
    #         orientation = "Back of Hand Toward Camera"
    #     else:
    #         orientation = "Side / Ambiguous"
    #     cv2.putText(image, orientation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #     size = getHandSize(hand_landmarks.landmark)
    #     updateHandSizeHistory(hand_size_history, size, time.time(), window_sec=hand_size_queue_window)
    #     cv2.putText(image, f"Hand size: {size:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    #     np_fig = update_plot(hand_size_history, window_sec=hand_size_queue_window, show=False)
    #     if np_fig is not None:
    #         overlay_numpy_fig(image, np_fig, alpha=0.5)

    #     zero_crossings = count_zero_crossings(hand_size_history, threshold=0.5)
    #     cv2.putText(image, f"Zero Crossings: {zero_crossings}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #     cv2.putText(image, f"Last Hand Orientation: {last_hand_orientation['orientation']}, Time: {last_hand_orientation['time']:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #     # test for gesture conditions
    #     is_hand_orientation_held = (last_hand_orientation['orientation']==hand_orientation) and (time.time()-last_hand_orientation['time'] >= hand_orientation_hold_window)
    #     action = 'Nothing'
    #     if is_hand_orientation_held and hand_orientation == HandOrientation.PALM_TOWARD and zero_crossings >= 2:
    #         print("go away action")
    #         action = 'Go Away'
    #     elif is_hand_orientation_held and hand_orientation == HandOrientation.BACK_OF_HAND and zero_crossings >= 2:
    #         print("come closer action")
    #         action = 'Come Closer'
    #     cv2.putText(image, f"Action: {action}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #     break


    # --------------------

    if FRAME_DRAWN is not None:
        cv2.imshow('Fist Movement Detection', FRAME_DRAWN)    
        frames.append(FRAME_DRAWN)
        timestamps.append(time.time()-start_time)
        FRAME_DRAWN = None

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hand_smoother.stop()

print("Saving video...")
fps = 60
frame_time = 1/fps
frame_durations = [t - s for s, t in zip(timestamps, timestamps[1:])]
frame_durations.append(frame_durations[-1])
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
next_frame = 1
running = frame_durations[0]
elapsed = 0
while next_frame < len(frame_durations):
    video.write(frames[next_frame-1])
    elapsed += frame_time
    if elapsed >= running:
        running += frame_durations[next_frame]
        next_frame += 1

video.release()