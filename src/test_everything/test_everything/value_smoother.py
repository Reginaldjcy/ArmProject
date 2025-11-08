import threading
import time

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
