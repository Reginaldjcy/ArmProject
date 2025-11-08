import numpy as np
import time
import cv2

class VideoCompositor:
    def __init__(self, fps, size, codec='mp4v'):
        self.fps = fps
        self.size = size
        self.codec = codec
        self.frames = []
        self.timestamps = []
    
    def add_frame(self, frame, timestamp_ms=None):
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)
        self.frames.append(frame)
        self.timestamps.append(timestamp_ms)
    
    def save_video(self, out_file):
        frame_time = 1000/self.fps
        frame_durations = [t - s for s, t in zip(self.timestamps, self.timestamps[1:])]
        frame_durations.append(frame_durations[-1])
        video = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*self.codec), self.fps, self.size)
        next_frame = 1
        running = frame_durations[0]
        elapsed = 0
        while next_frame < len(frame_durations):
            video.write(self.frames[next_frame-1])
            elapsed += frame_time
            if elapsed >= running:
                running += frame_durations[next_frame]
                next_frame += 1
        video.release()
    
