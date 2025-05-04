import os
import torch

model_path = os.path.join(os.path.dirname(__file__), "hopenet_robust_alpha1.pkl")
print("Model path:", model_path)

model = torch.load(model_path, map_location='cpu')
model.eval()
print("Model loaded!")
