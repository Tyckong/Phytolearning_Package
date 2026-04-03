from roboflow import Roboflow
from ultralytics import YOLO                        #On this first iteration I'll test YOLO26 alone on itself, just to test things.
import os

#Dataset used for the test training in my computer. 
rf = Roboflow(api_key="vxn2kEr9CvN9YtaYLSxI")
project = rf.workspace("tomss-workspace-woalo").project("rice-stomata-1unfk")
version = project.version(1)
dataset = version.download("yolo26")

print(f"Dataset downloaded in: {dataset.location}")
print("--[x]--{ Loading base architecture for YOLO26 }--[x]--")
model = YOLO('yolo26n.pt')

#Ah yes, Micro-Training begins now B)
print("--[x]--{ Initiating Local Micro-Training }--[x]--")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=10,
    imgsz=640,
    batch=4,
    workers=2,
    device='cpu',
    project='Phytolearning_Tests',
    name='test_50_imgs_yolo26'
)

#Testing that the model can receive a new image and output a prediction.
print("--[x]--{ Initiating Test Prediction }--[x]--")
test_img_path = f"{dataset.location}/valid/images"  #Path to validation images.
if os.path.exists(test_img_path):
    test_imgs = os.listdir(test_img_path)
    if test_imgs:
        selection = os.path.join(test_img_path, test_imgs[0])       #Take the first availabe image.
        predition = model.predict(selection, save=True)
        print("Prediction saved on runs/Phytolearning_Tests/test_50_imgs_yolo26")