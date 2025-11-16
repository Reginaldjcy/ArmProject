import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

import torch
import numpy as np
from scipy.spatial.distance import cdist

class GlobalReIDMemory:
    def __init__(self, sim_threshold=0.7, max_entries=2000):
        self.memory = {}  # id -> embedding (numpy)
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries

    def add(self, track_id, emb):
        if emb is None:
            return
        # 🔹 自动兼容 torch / numpy
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy().astype(np.float32)
        else:
            emb_np = emb.astype(np.float32)
        self.memory[track_id] = emb_np
        if len(self.memory) > self.max_entries:
            self.memory.pop(next(iter(self.memory)))

    def match(self, emb):
        if not self.memory:
            return None, 0.0
        if isinstance(emb, torch.Tensor):
            emb_np = emb.detach().cpu().numpy().astype(np.float32)
        else:
            emb_np = emb.astype(np.float32)
        keys = list(self.memory.keys())
        feats = np.stack([self.memory[k] for k in keys])
        sims = 1 - cdist([emb_np], feats, metric='cosine')
        idx = np.argmax(sims)
        sim = sims[0, idx]
        if sim >= self.sim_threshold:
            return keys[idx], sim
        return None, sim


#################################3
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load detector with pretrained weights and preprocessing transforms
weights = Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
detector.to(device).eval()
transform = weights.transforms()

# Initialize tracker
tracker = BoostTrack(reid_weights=Path('lmbn_n_cuhk03_d.pt'), device=device, half=False)

global_reid = GlobalReIDMemory(sim_threshold=0.7)

# Start video capture
cap = cv2.VideoCapture(0)

with torch.inference_mode():
    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
        input_tensor = transform(tensor).to(device)

        output = detector([input_tensor])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= 0.5
        boxes = output['boxes'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        filtered_scores = scores[keep]
        is_person = labels == 1
        boxes = boxes[is_person]
        filtered_scores = filtered_scores[is_person]
        labels = labels[is_person]

        detections = np.concatenate(
            [boxes, filtered_scores[:, None], labels[:, None]],
            axis=1
        )

        # 🔹 更新 tracker
        res = tracker.update(detections, frame)

        # 🔹 维护 ReID embedding 缓存
        for trk in tracker.active_tracks:
            if trk.time_since_update == 0 and trk.hit_streak >= tracker.min_hits:
                emb = trk.get_emb()  # 最新特征
                global_reid.add(trk.id, emb)

        # 🔹 检查新目标是否和旧的ID相似
        for trk in tracker.active_tracks:
            if trk.age < 3:  # 只检测刚出现的目标
                emb = trk.get_emb()
                old_id, sim = global_reid.match(emb)
                if old_id and old_id != trk.track_id:
                    print(f"Reassign {trk.track_id} -> {old_id} (sim={sim:.2f})")
                    trk.track_id = old_id  # ✅ 重用旧 ID

        tracker.plot_results(frame, show_trajectories=True)
        cv2.imshow('BoXMOT + Global ReID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
