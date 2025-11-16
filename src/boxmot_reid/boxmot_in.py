import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" if multi-GPU

import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights as Weights
)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load detector with pretrained weights and preprocessing transforms
weights = Weights.DEFAULT
detector = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
detector.to(device).eval()
transform = weights.transforms()

# Initialize tracker
tracker = BoostTrack(
    reid_weights=Path('osnet_x1_0_msmt17.pt'),  # 替换成更强 ReID 模型
    device='0',
    half=False,
    with_reid=True,
    max_age=180,       # 延长跟踪存活时间 (默认 60)
    min_hits=3,        # 稍微提高确认门槛
    lambda_iou=0.3,
    lambda_mhd=0.2,
    lambda_shape=0.5,  # 稍微增强形状一致性
)

# Start video capture
cap = cv2.VideoCapture(0)

with torch.inference_mode():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB and prepare for detector
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
        input_tensor = transform(tensor).to(device)

        # Run detection
        output = detector([input_tensor])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= 0.5

        # Prepare detections for tracking
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
        # Update tracker and draw results
        #   INPUT:  M X (x, y, x, y, conf, cls)
        #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
        res = tracker.update(detections, frame)
        tracker.plot_results(frame, show_trajectories=True)

        # Show output
        cv2.imshow('BoXMOT + Torchvision', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()