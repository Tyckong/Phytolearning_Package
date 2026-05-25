import os
import torch
from roboflow import Roboflow
from ultralytics import YOLO

# ===[ YOLO26 ]===
print("--[x]--{ Preparing Roboflow Dataset }--[x]--")
rf = Roboflow(api_key="vxn2kEr9CvN9YtaYLSxI")
project = rf.workspace("tomss-workspace-woalo").project("stomata-pore-detection-unified")
dataset = project.version(1).download("yolov8")

print(f"Dataset downloaded in: {dataset.location}")
print("--[x]--{ Loading base architecture for YOLO26 }--[x]--")
model = YOLO('yolo26n.pt')

#Ah yes, Micro-Training begins now B)
print("--[x]--{ Initiating Local Microtraining }--[x]--")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=15,
    imgsz=640,
    batch=4,
    workers=2,
    device='cpu',
    project='Phytolearning_Tests',
    name='test_jerarquico',
    exist_ok=True
)