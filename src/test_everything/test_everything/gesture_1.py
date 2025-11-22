'''

HOLD: right hand up, right top of body
UP: swipe from chest height up with right hand
DOWN: swipe from chest height down with right hand
LEFT: swipe from center height left with left hand
RIGHT: swipe from center height right with right hand

'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import bisect
import time
import os

# 你本地的模块（路径按你的工程实际调整）
from ...basic_function.basic_function.gesture_class import ValueSmoother
from .calc_pose_landmarks import PoseLandmarkerSolver
from .video_compositor import VideoCompositor

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

FRAME_IMAGE = None
FRAME_DRAWN = None

SWIPE_TIME_THRESHOLD = 300

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
HOLD_TIME_THRESHOLD = 1000
HOLD_START_XY_HISTORY = []
HOLD_START_TIME_HISTORY = []

DISPLAY_TEXT = ''
DISPLAY_TIME = 0
DISPLAY_TIME_THRESHOLD = 500

def handle_left_swipe(lm, thread_time):
    global SWIPE_TIME_THRESHOLD
    global LEFT_START, LEFT_START_TIME, LEFT_PREVIOUS_X
    
    try:
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_x_dist = np.abs(shoulder_left[0] - shoulder_right[0])

        wrist_left = lm[15][:2]

        if  wrist_left[0] < shoulder_left[0] and wrist_left[0] > shoulder_right[0]:
            if not LEFT_START:
                # print('start left swipe')
                LEFT_START = True
            LEFT_START_TIME = thread_time
            LEFT_PREVIOUS_X = wrist_left[0]
        
        if not LEFT_START: return

        if thread_time - LEFT_START_TIME > SWIPE_TIME_THRESHOLD:
            # print('timeout left swipe')
            LEFT_START = False
        
        if wrist_left[0] < LEFT_PREVIOUS_X:
            # print('stop left swipe')
            LEFT_START = False
        
        if wrist_left[0] > shoulder_left[0]+shoulder_x_dist:
            # print('end left swipe')
            LEFT_START = False
            return True
        
        LEFT_PREVIOUS_X = wrist_left[0]
    except Exception as e:
        print("Error in handle_left_swipe:", e)
    return False

def handle_right_swipe(lm, thread_time):
    global SWIPE_TIME_THRESHOLD
    global RIGHT_START, RIGHT_START_TIME, RIGHT_PREVIOUS_X

    try:
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_x_dist = np.abs(shoulder_left[0] - shoulder_right[0])

        wrist_right = lm[16][:2]

        if  wrist_right[0] < shoulder_left[0] and wrist_right[0] > shoulder_right[0]:
            if not RIGHT_START:
                # print('start right swipe')
                RIGHT_START = True
            RIGHT_START_TIME = thread_time
            RIGHT_PREVIOUS_X = wrist_right[0]
        
        if not RIGHT_START: return

        if thread_time - RIGHT_START_TIME > SWIPE_TIME_THRESHOLD:
            # print('timeout right swipe')
            RIGHT_START = False
        
        if wrist_right[0] > RIGHT_PREVIOUS_X:
            # print('stop right swipe')
            RIGHT_START = False
        
        if wrist_right[0] < shoulder_right[0]-shoulder_x_dist:
            # print('end right swipe')
            RIGHT_START = False
            return True
        
        RIGHT_PREVIOUS_X = wrist_right[0]
    except Exception as e:
        print("Error in handle_right_swipe:", e)
    return False

def handle_down_swipe(lm, thread_time):
    global SWIPE_TIME_THRESHOLD
    global DOWN_START, DOWN_START_TIME, DOWN_START_Y

    try:
        nose = lm[0][:2]
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1]+shoulder_right[1])/2
        bound_y_dist = np.abs(shoulder_y - nose[1])
        upper_bound = shoulder_y+bound_y_dist/2
        upper_bound_complete = upper_bound+bound_y_dist/2*3

        wrist = lm[16][:2] 

        if nose[1] < wrist[1] < upper_bound:
            if not DOWN_START:
                DOWN_START = True
            DOWN_START_TIME = thread_time
            DOWN_START_Y    = wrist[1]

        if not DOWN_START:
            return False

        if thread_time - DOWN_START_TIME > SWIPE_TIME_THRESHOLD:
            DOWN_START = False
            return False

        if wrist[1] < DOWN_START_Y:
            DOWN_START = False
            return False

        if wrist[1] > upper_bound_complete:
            DOWN_START = False
            return True

        DOWN_START_Y = wrist[1]

    except Exception as e:
        print("Error in handle_down_swipe:", e)
    return False

def handle_up_swipe(lm, thread_time):
    global SWIPE_TIME_THRESHOLD
    global UP_START, UP_START_TIME, UP_START_Y

    try:
        nose = lm[0][:2]
        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1]+shoulder_right[1])/2
        bound_y_dist = np.abs(shoulder_y - nose[1])
        upper_bound = shoulder_y+bound_y_dist/2
        lower_bound_complete = nose[1]-bound_y_dist/2*3

        wrist = lm[16][:2]

        if nose[1] < wrist[1] < upper_bound:
            if not UP_START:
                UP_START = True
            UP_START_TIME = thread_time
            UP_START_Y    = wrist[1]

        if not UP_START:
            return False

        if thread_time - UP_START_TIME > SWIPE_TIME_THRESHOLD:
            UP_START = False
            return False

        if wrist[1] > UP_START_Y:
            UP_START = False
            return False

        if wrist[1] < lower_bound_complete:
            UP_START = False
            return True

        UP_START_Y = wrist[1]

    except Exception as e:
        print("Error in handle_up_swipe:", e)
    return False

def handle_hold(lm, thread_time):
    global HOLD_START
    global HOLD_TIME_THRESHOLD
    global HOLD_START_XY_HISTORY, HOLD_START_TIME_HISTORY

    global CAMERA_WIDTH, CAMERA_HEIGHT

    try:
        wrist = lm[16][:2]

        shoulder_left = lm[11][:2]
        shoulder_right = lm[12][:2]
        shoulder_y = (shoulder_left[1]+shoulder_right[1])/2

        if wrist[1] < shoulder_y and wrist[0] < shoulder_right[0]:
            HOLD_START = True
        else:
            HOLD_START = False
            HOLD_START_XY_HISTORY = []
            HOLD_START_TIME_HISTORY = []
        
        if not HOLD_START:
            return False

        HOLD_START_XY_HISTORY.append(wrist)
        HOLD_START_TIME_HISTORY.append(thread_time)

        hold_start_time = thread_time-HOLD_TIME_THRESHOLD
        if hold_start_time < HOLD_START_TIME_HISTORY[0]:
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

        window_size_x = (window_max_x - window_min_x)*CAMERA_WIDTH
        window_size_y = (window_max_y - window_min_y)*CAMERA_HEIGHT

        # use shoulder dist as reference
        dx = (shoulder_left[0] - shoulder_right[0]) * CAMERA_WIDTH
        dy = (shoulder_left[1] - shoulder_right[1]) * CAMERA_HEIGHT
        shoulder_distance_px = np.sqrt(dx**2 + dy**2)

        window_threshold = shoulder_distance_px * 0.1

        if window_size_x < window_threshold and window_size_y < window_threshold:
            return True

    except Exception as e:
        print("Error in handle_hold:", e)
    return False

def pose_smoother_callback(values, thread_time, raw_timestamp, og_vars):
        global FRAME_IMAGE, FRAME_DRAWN
        global DISPLAY_TEXT, DISPLAY_TIME, DISPLAY_TIME_THRESHOLD
        if FRAME_IMAGE is None:
            return
        FRAME_DRAWN = FRAME_IMAGE.copy()
        try:
            lm = np.array(values).reshape(-1, 3)
            pose_landmarks = lm[0:33]
            pose_world_landmarks = lm[33:66]
            # for i in range(33):
            #     if i in [11, 12, 15, 16]:
            #         cv2.circle(FRAME_DRAWN, (int(pose_landmarks[i][0]*FRAME_DRAWN.shape[1]), int(pose_landmarks[i][1]*FRAME_DRAWN.shape[0])), 5, (255, 0, 0), -1)
            #     else:
            #         cv2.circle(FRAME_DRAWN, (int(pose_landmarks[i][0]*FRAME_DRAWN.shape[1]), int(pose_landmarks[i][1]*FRAME_DRAWN.shape[0])), 3, (0, 0, 255), -1)
            shoulder_left = pose_landmarks[11][:2]
            shoulder_right = pose_landmarks[12][:2]
            shoulder_x_dist = np.abs(shoulder_left[0] - shoulder_right[0])

            nose = pose_landmarks[0][:2]
            shoulder_y = (shoulder_left[1]+shoulder_right[1])/2
            bound_y_dist = np.abs(shoulder_y - nose[1])
            upper_bound = shoulder_y+bound_y_dist/2

            # cv2.line(FRAME_DRAWN, (0, int(shoulder_y*FRAME_DRAWN.shape[0])), (FRAME_DRAWN.shape[1], int(shoulder_y*FRAME_DRAWN.shape[0])), (0, 255, 0), thickness=2)
            # cv2.line(FRAME_DRAWN, (int(shoulder_right[0]*FRAME_DRAWN.shape[1]), 0), (int(shoulder_right[0]*FRAME_DRAWN.shape[1]), FRAME_DRAWN.shape[0]), (0, 255, 255), thickness=2)

            is_left_swipe = handle_left_swipe(pose_landmarks, thread_time)
            is_right_swipe = handle_right_swipe(pose_landmarks, thread_time)
            is_down_swipe = handle_down_swipe(pose_landmarks, thread_time)
            is_up_swipe = handle_up_swipe(pose_landmarks, thread_time)

            is_hold = handle_hold(pose_landmarks, thread_time)

            if thread_time - DISPLAY_TIME > DISPLAY_TIME_THRESHOLD:
                DISPLAY_TEXT = ''
            if is_left_swipe:
                DISPLAY_TIME = thread_time
                DISPLAY_TEXT = 'left swipe'
            if is_right_swipe:
                DISPLAY_TIME = thread_time
                DISPLAY_TEXT = 'right swipe'
            if is_down_swipe:
                DISPLAY_TIME = thread_time
                DISPLAY_TEXT = 'down swipe'
            if is_up_swipe:
                DISPLAY_TIME = thread_time
                DISPLAY_TEXT = 'up swipe'
            if is_hold:
                DISPLAY_TIME = thread_time
                DISPLAY_TEXT = 'hold'
            cv2.putText(FRAME_DRAWN, DISPLAY_TEXT, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

if __name__ == "__main__":

    pose_landmark_smoother = ValueSmoother(pose_smoother_callback, alpha=0.2, target_fps=60)
    pose_landmark_smoother.start()

    pose_landmark_solver = PoseLandmarkerSolver(lambda res,img,time:pose_landmarker_callback(pose_landmark_smoother,res,img,time), "pose_landmarker_full.task")

    print("Starting Webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    videoCompositor = VideoCompositor(60, (CAMERA_WIDTH, CAMERA_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            print('error reading frame')
            break

        FRAME_IMAGE = frame.copy()
        pose_landmark_solver.solve_async(frame)

        if FRAME_DRAWN is not None:
            videoCompositor.add_frame(FRAME_DRAWN, int(time.time() * 1000))
            cv2.imshow(".", FRAME_DRAWN)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('breaking...')
            break

    print('exiting...')
    cv2.destroyAllWindows()

    pose_landmark_solver.close()
    pose_landmark_smoother.stop()
    pose_landmark_smoother.join()

    cap.release()
    videoCompositor.save_video('output.mp4')