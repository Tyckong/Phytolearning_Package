import os
import cv2                                                  #Load OpenCV (Open Source Computer Vision) library. This is some fascinating stuff.
import torch                                                #Loads PyTorch, open source ML library. I've used this before and, somehow, had never realized what it was. Funny.
import numpy as np
import random
from roboflow import Roboflow
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

# ===[ YOLO26 ]===
print("--[x]--{ Preparing Roboflow Dataset }--[x]--")
rf = Roboflow(api_key="vxn2kEr9CvN9YtaYLSxI")
project = rf.workspace("tomss-workspace-woalo").project("stomata-pore-detection-unified")
dataset = project.version(1).download("yolov8")

#Testing that the model can receive a new image and output a prediction.
print("--[x]--{ Initiating Test Prediction }--[x]--")
test_img_path = f"{dataset.location}/valid/images"          #Path to validation images.
test_imgs = os.listdir(test_img_path)
if not test_imgs:
    raise FileNotFoundError("No validation images found on the dataset.")
IMG_PATH = os.path.join(test_img_path, random.choice(test_imgs))

#Here lay the weight paths.
YOLO_WEIGHTS = "runs/detect/Phytolearning_Tests/test_jerarquico/weights/best.pt"
MOBILE_SAM_WEIGHTS = "mobile_sam.pt"

#Loading info on memory.
yolo_model_best = YOLO(YOLO_WEIGHTS)
sam_model = sam_model_registry["vit_t"](checkpoint = MOBILE_SAM_WEIGHTS)
sam_model.to(device = "cpu")
sam_predictor = SamPredictor(sam_model)

#Now, we process the image.
img_bgr = cv2.imread(IMG_PATH)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("\n--[x]--{ Phase 1: Stoma Detection (YOLO26) }--[x]--")
yolo_results = yolo_model_best.predict(IMG_PATH, conf=0.25, verbose=False)
boxes = yolo_results[0].boxes.xyxy.cpu().numpy()

print(f"{len(boxes)} Stomas detected.")

print("\n--[x]--{ Phase 2: Cropping Morphometric Analysis and Visuals  }--[x]--")
THRESHOLD_CLOSED_PIXELS = 100    

open_stoma = 0
closed_stoma = 0

color_mask_open = np.array([0, 255, 0, 128])            #Color the masks over the open stoma so we can actually see them lol
color_mask_closed = np.array([0, 0, 255, 128])          #Same, but for closed ones.
output_img = img_bgr.copy()

for box in boxes:                                       #Check stoma by stoma.
    x1, y1, x2, y2 = map(int, box)

    h_img, w_img = img_rgb.shape[:2]                    #Crop the original image. We'll add a small margin so we don't cut the borders of the stoma.
    cx1 = max(0, x1 - 5)
    cy1 = max(0, y1 - 5)
    cx2 = min(w_img, x2 + 5)
    cy2 = min(h_img, y2 + 5)

    crop_rgb = img_rgb[cy1:cy2, cx1:cx2]
    crop_h, crop_w = crop_rgb.shape[:2]

    sam_predictor.set_image(crop_rgb)                   #MobileSAM analyzes ONLY the cropped image.

    input_point = np.array([[crop_w // 2, crop_h // 2]])#Look at the center of the cropped image.
    input_label = np.array([1])                         #Labels it as an interest point... in other words: the pore.

    masks, scores, logits = sam_predictor.predict(
        point_coords = input_point,
        point_labels = input_label,
        multimask_output = True,
    )

    areas = [np.sum(masks[j]) for j in range(masks.shape[0])]
    valid_areas = [a for a in areas if a > 10]          #Filters the smallest valid mask inside the crop.

    if valid_areas:
        target_area = min(valid_areas)
        pore_index = areas.index(target_area)
    else:
        pore_index = np.argmax(areas)
    
    best_mask_crop = masks[pore_index]
    pixel_area = np.sum(best_mask_crop)

    full_mask = np.zeros((h_img, w_img), dtype=bool)    #Create an empty mask of the original image size.
    full_mask[cy1:cy2, cx1:cx2] = best_mask_crop

    if pixel_area < THRESHOLD_CLOSED_PIXELS:
        closed_stoma += 1
        color_box = (0, 0, 255)
        output_img[full_mask] = (output_img[full_mask]*0.5)+(color_mask_closed[:3]*0.5)
    else:
        open_stoma += 1
        color_box = (255, 0, 0)
        output_img[full_mask] = (output_img[full_mask]*0.5)+(color_mask_open[:3]*0.5)
    
    cv2.rectangle(output_img, (x1, y1), (x2, y2), color_box, 2)
    cv2.putText(output_img, f"{pixel_area}px", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)

cv2.imwrite("yolo_sam_result.jpg", output_img)
print("Masks applied on crops, areas calculated and saved to 'yolo_sam_result.jpg'.")

total_stomas = len(boxes)

if total_stomas > 0:
    closed_pct = (closed_stoma / total_stomas) * 100

    if closed_pct <= 20:
        veredict = "Planta Sana (Control)"
    elif closed_pct <= 60:
        veredict = "Estrés Hídrico Moderado"
    else:
        veredict = "Estrés Hídrico Severo (Sequía)"

    print("\n===[x]==={ Final Morphometric Results }===[x]===")
    print(f"Total de Estomas Analizados: {total_stomas}")
    print(f"Estomas Abiertos (> {THRESHOLD_CLOSED_PIXELS}px): {open_stoma}")
    print(f"Estomas Cerrados (< {THRESHOLD_CLOSED_PIXELS}px): {closed_stoma}")
    print(f"Porcentaje de Cierre Estomático: {closed_pct:.2f}%")
    print(f"Veredicto Final: {veredict}")
else:
    print("No stomas were detected. Cannot perform morphometric analysis.")

print("\n Results ready, boss.")