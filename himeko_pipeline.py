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

"""print(f"Dataset downloaded in: {dataset.location}")
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
) """

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
yolo_results = yolo_model_best.predict(IMG_PATH, conf=0.2, verbose=False)
boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
classes = yolo_results[0].boxes.cls.cpu().numpy()

stomas_idx = np.where(classes == 0)[0]
pores_idx = np.where(classes == 1)[0]

stoma_boxes = boxes[stomas_idx]
pore_boxes = boxes[pores_idx]

print(f"{len(stoma_boxes)} Stoma and {len(pore_boxes)} Pores detected.")

print("\n--[x]--{ Phase 2: Drawing Masks with MobileSAM }--[x]--")
sam_predictor.set_image(img_rgb)  #Here, TinyViT scans the whole image, calculates the patches and generates the embedding.

pore_masks = []
if len(pore_boxes) > 0:
    input_boxes = torch.tensor(pore_boxes, device=sam_model.device)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(input_boxes, img_rgb.shape[:2])

    masks, _, _ = sam_predictor.predict_torch(    #Sam draws the fine borders of all the pores based on the YOLO boxes.
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )
    
    for i in range(masks.shape[0]):                 #This is to select the smaller mask from the bunch due to problems where SAM chooses the entire stoma.
        options = masks[i]
        areas = [torch.sum(options[j]).item() for j in range(options.shape[0])]
        valid_areas = [a for a in areas if a>10]    #Filter masks that are just noise. In this case, masks over 10 pixels, kinda arbitrary, but we'll see.

        if valid_areas:
            target_area = min(valid_areas)
            pore_index = areas.index(target_area)
        else:                                       #If all of them are smaller than 10px, we'll choose the bigger one.
            pore_index = np.argmax(areas)

        isolated_mask = options[pore_index].unsqueeze(0)
        pore_masks.append(isolated_mask)

print("\n--[x]--{ Phase 3: Morphometric Analysis and Visuals }--[x]--")
THRESHOLD_CLOSED_PIXELS = 100    

open_stoma = 0
closed_stoma = 0

color_mask_open = np.array([0, 255, 0, 128])            #Color the masks over the open stoma so we can actually see them lol
color_mask_closed = np.array([0, 0, 255, 128])          #Same, but for closed ones.
output_img = img_bgr.copy()

processed_pores = set()                                 #Memory to track already used pores.

for s_box in stoma_boxes:                               #Check stoma by stoma.
    sx1, sy1, sx2, sy2 = s_box
    
    pore_found_inside = False
    assigned_pore_idx = -1
    
    for p_idx, p_box in enumerate(pore_boxes):          #Look if there's any other pore whose center is within this stoma.
        px1, py1, px2, py2 = p_box
        pore_center_x = (px1 + px2) / 2
        pore_center_y = (py1 + py2) / 2
        
        if sx1-10 <= pore_center_x <= sx2+10 and sy1-10 <= pore_center_y <= sy2+10:
            pore_found_inside = True
            assigned_pore_idx = p_idx
            processed_pores.add(p_idx)                  #Save pore in memory.
            break                                       #Stoma's pore found.
            
    if pore_found_inside:                               #If pore found -> Color mask and measure.
        mask_numpy = pore_masks[assigned_pore_idx][0].cpu().numpy()      
        pixel_area = np.sum(mask_numpy)         

        if pixel_area < THRESHOLD_CLOSED_PIXELS:
            closed_stoma += 1
            color_box = (0, 0, 255)
            output_img[mask_numpy] = (output_img[mask_numpy]*0.5) + (color_mask_closed[:3]*0.5)
        else:
            open_stoma += 1
            color_box = (255, 0, 0)
            output_img[mask_numpy] = (output_img[mask_numpy]*0.5) + (color_mask_open[:3]*0.5)

        cv2.rectangle(output_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color_box, 2) #Color depending on state.
        cv2.putText(output_img, f"{pixel_area}px", (int(sx1), int(sy1)-5),                  #Area measurments text.
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)
        
    else:                                               #If no pore is found, we assume it's closed.
        closed_stoma += 1
        color_box = (0, 0, 255)
        cv2.rectangle(output_img, (int(sx1), int(sy1)), (int(sx2), int(sy2)), color_box, 2)
        cv2.putText(output_img, "0px (Cerrado)", (int(sx1), int(sy1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)

for p_idx, p_box in enumerate(pore_boxes):              #Evaluate pores without any stoma detected.
    if p_idx not in processed_pores:
        px1, py1, px2, py2 = p_box
        mask_numpy = pore_masks[p_idx][0].cpu().numpy()
        pixel_area = np.sum(mask_numpy)

        if pixel_area < THRESHOLD_CLOSED_PIXELS:
            closed_stoma += 1
            color_box = (0, 0, 255)
            output_img[mask_numpy] = (output_img[mask_numpy]*0.5) + (color_mask_closed[:3]*0.5)
        else:
            open_stoma += 1
            color_box = (255, 0, 0)
            output_img[mask_numpy] = (output_img[mask_numpy]*0.5) + (color_mask_open[:3]*0.5)

        cv2.rectangle(output_img, (int(px1), int(py1)), (int(px2), int(py2)), color_box, 2)
        cv2.putText(output_img, f"{pixel_area}px", (int(px1), int(py1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1)

cv2.imwrite("yolo_sam_result.jpg", output_img)
print("Masks applied, areas calculated and saved to 'yolo_sam_result.jpg'.")

total_stomas_analizados = len(stoma_boxes) + (len(pore_boxes) - len(processed_pores))

if total_stomas_analizados > 0:
    closed_pct = (closed_stoma / total_stomas_analizados) * 100

    if closed_pct <= 20:
        veredict = "Healthy Plant // Planta Sana (Control)"
    elif closed_pct <= 60:
        veredict = "Mild Drought State // Estrés Hídrico Moderado"
    else:
        veredict = "Severe Drought State // Estrés Hídrico Severo (Sequía)"

    print("\n===[x]==={ Final Morphometric Results }===[x]===")
    print(f"Total de Estomas Analizados: {total_stomas_analizados}")
    print(f"Estomas Abiertos (> {THRESHOLD_CLOSED_PIXELS}px): {open_stoma}")
    print(f"Estomas Cerrados (Sin poro o < {THRESHOLD_CLOSED_PIXELS}px): {closed_stoma}")
    print(f"Porcentaje de Cierre Estomático: {closed_pct:.2f}%")
    print(f"Veredicto Final: {veredict}")
else:
    print("No stomas were detected. Cannot perform morphometric analysis.")

print("\n Results ready, boss.")