import os
import cv2                                                              #Load OpenCV (Open Source Computer Vision) library. This is to manipulate images mathematically.
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))                 #Dynamic route to the script's directory.

#Here lay the fabled weight paths.
YOLO_WEIGHTS = os.path.join(MODULE_DIR, "weights", "yolo26s.pt")
MOBILE_SAM_WEIGHTS = os.path.join(MODULE_DIR, "weights", "mobile_sam.pt")

#Hyperparameter tunes obtained by using Optuna.
CONF_STOMA = 0.132                                                      #To detect stoma even if the net is not 100% sure.
CONF_PORE_BASE = 0.615                                                  #Filters excesive noise of the predicted pores.
PADDING_YOLO = 30
PADDING_SAM = 0

def analyze_images(input_path, output_dir, yolo_weights=YOLO_WEIGHTS, sam_weights=MOBILE_SAM_WEIGHTS, aperture_th=12.5, healthy_th=25.0, moderate_th=75.0):    
    print("--[x]--{ Initiating Phytolearning Inference }--[x]--")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path '{input_path}' not found")
    
    os.makedirs(output_dir, exist_ok=True)
    
    #Single file or Directory detection for single or multiple image analysis respectively.
    image_paths = []
    if os.path.isfile(input_path):                                      #If the input is a single image...
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_paths.append(input_path)                              #We append it to the list as long as it is a valid image file.
        else:
            raise ValueError("The input file is not a valid image type.")
    elif os.path.isdir(input_path):                                     #If it is a directory...
        image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]    #Add all the valid image files inside of it into the list.
        
    if not image_paths:
        raise FileNotFoundError("No valid images found in the specified route.")
    
    print("\n[+] Loading YOLO26s and MobileSAM weights...")
    if not os.path.exists(yolo_weights):
        raise FileNotFoundError(f"{yolo_weights} file missing.")
    if not os.path.exists(sam_weights):
        raise FileNotFoundError(f"{sam_weights} file missing.")
        
    yolo_model_best = YOLO(yolo_weights)                                #Loads the weights of the trained YOLO model.
    sam_model = sam_model_registry["vit_t"](checkpoint=sam_weights)     #Loads the MobileSAM weights.
    sam_model.to(device="cpu")                                          #Tells it to use the machine's cpu.
    sam_predictor = SamPredictor(sam_model)                             #Ensambles the SAM predictor.

    #Batch iteration logic.
    for img_path in image_paths:                                        #Iterate through all the found images.
        img_name = os.path.basename(img_path)                           #Save the image's name just so the user can debug easier in case something happens.
        print(f"\n[+] Analyzing: {img_name}")
        
        img_bgr = cv2.imread(img_path)                                  #Process the image's bgr format...
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)              #...into rgb format because most most libraries use that format for some reason.
        h_img, w_img = img_rgb.shape[:2]                                #We ask YOLO to save the height and width.

        yolo_results = yolo_model_best.predict(
            source=img_path,
            conf=CONF_STOMA,                                            #Confidence Threshold. 
            iou=0.30,                                                   #NMS threshold. 
            agnostic_nms=True,                                          #Useful if the unified dataset created "ghost classes".
            verbose=False                                               #Mutes YOLO's default prints to keep the console clean.
        )

        boxes_data = yolo_results[0].boxes                              #Saves the coordinates of the generated boxes.

        #Split the boxes into two categories.
        stoma_boxes = [b for b in boxes_data if int(b.cls[0].item()) == 0]
        all_pores = [(b.xyxy[0].cpu().numpy(), b.conf[0].item()) for b in boxes_data if int(b.cls[0].item()) == 1]

        print(f"Detected {len(stoma_boxes)} Stomas and {len(all_pores)} Pores.")

        open_stoma = 0                                                  #Open Stoma Counter.
        closed_stoma = 0                                                #Closed Stoma Counter.
        color_mask_open = np.array([0, 255, 0, 128])                    #Color the masks over the open stoma so we can actually see them lol
        color_mask_closed = np.array([0, 0, 255, 128])                  #Same, but for closed ones.
        output_img = img_bgr.copy()                                     #Copies the bgr image to later ensamble the masks onto it.

        if len(stoma_boxes) == 0:
            print(f"WARNING: No se detectaron estomas en {img_name}")
            continue

        for s_box in stoma_boxes:                                       #Check stoma by stoma.
            sx1, sy1, sx2, sy2 = map(int, s_box.xyxy[0].cpu().numpy())  #Fetch stoma coordinates.
            stoma_area_px = (sx2 - sx1) * (sy2 - sy1)                   #Calculate the área of the stoma.

            pores_inside = []                                           #Here we'll save all detected pores within the image.
            for p_box, p_conf in all_pores:                             #Iterate through all the pore bounding boxes detected.
                px1, py1, px2, py2 = map(int, p_box)                    #Get their coordinates.
                if max(sx1, px1) < min(sx2, px2) and max(sy1, py1) < min(sy2, py2): #Check if the bounding boxes intersect.
                    pores_inside.append((px1, py1, px2, py2, p_conf))   #If they do, append them.

            matched_pore = None
            if len(pores_inside) > 0:
                high_conf_pores = [p for p in pores_inside if p[4] >= CONF_PORE_BASE] #Filters the pores below the base confidence.
                if len(high_conf_pores) > 0:                            #If any exists...
                    matched_pore = max(high_conf_pores, key=lambda p: p[4])[:4] #...Pick the one with the highest confidence.
                else:                                                   #If none exists...
                    matched_pore = max(pores_inside, key=lambda p: p[4])[:4]    #...Stick to the one with the highest confidence from the ones found.

            #Add padding to the bounding boxes.
            cx1, cy1 = max(0, sx1 - PADDING_YOLO), max(0, sy1 - PADDING_YOLO)
            cx2, cy2 = min(w_img, sx2 + PADDING_YOLO), min(h_img, sy2 + PADDING_YOLO)
            
            crop_rgb = img_rgb[cy1:cy2, cx1:cx2]                    #Crop the stoma complex.
            crop_h, crop_w = crop_rgb.shape[:2]                     #Obtain the measurments of the crop.

            if crop_h == 0 or crop_w == 0:                          #If by any chance, the crop has no pixels, discard this iteration.
                continue

            #Apply CLAHE to the crops, turning them into black and white images for better detail recognition.
            lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB)         #Pass the crop to lab.
            l_channel, a_channel, b_channel = cv2.split(lab)        #Separate luminosity from lab.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #Pass the result through CLAHE.
            cl = clahe.apply(l_channel)                             #Apply adaptative contrast only to the luminosity so the pore is easier to see against the background.
            limg = cv2.merge((cl, a_channel, b_channel))            #Merge the channels.
            crop_clahe_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  #Go back to rgb so MobileSAM can receive a high quality processed image.

            sam_predictor.set_image(crop_clahe_rgb)                 #Hand the B&W image to the SAM predictor.

            if matched_pore is not None:                            
                #Open stoma. Hybrid prompting: Point and Box prompting >:)
                px1, py1, px2, py2 = matched_pore
                
                #Translate absolute coordinates of the pore to the relative coordinates of the crop.
                rel_px1, rel_py1 = max(0, px1 - cx1), max(0, py1 - cy1)
                rel_px2, rel_py2 = min(crop_w, px2 - cx1), min(crop_h, py2 - cy1)
                
                rel_cx = (rel_px1 + rel_px2) // 2                   
                rel_cy = (rel_py1 + rel_py2) // 2                   

                point_coords = np.array([[rel_cx, rel_cy]])         #Give it the coordinates of the pore's center.
                point_labels = np.array([1])                        #Point prompt.
                
                input_box = np.array([                              #And also give it the coordinates to the box of the pore, slightly expanded by pad so the borders are within the area.
                    max(0, rel_px1 - PADDING_SAM), max(0, rel_py1 - PADDING_SAM),
                    min(crop_w, rel_px2 + PADDING_SAM), min(crop_h, rel_py2 + PADDING_SAM)
                ])

                masks, scores, _ = sam_predictor.predict(           #Sam prediction's execution.
                    point_coords=point_coords, 
                    point_labels=point_labels,
                    box=input_box[None, :],                         #Box prompt.
                    multimask_output=True,                          #Generate 3 options.
                )
            else:                                                   #YOLO found a stoma, cropped the image, but was unable to detect a bounding box for the pore. In this case, we give the image to MobileSAM so it can detect at least something.
                #Since we don't have a bounding box to guide MobileSAM, we'll give the model the center of the image as a loose prompt.
                rel_cx = crop_w // 2
                rel_cy = crop_h // 2
                
                point_coords = np.array([[rel_cx, rel_cy]])         #Coordinates of the center.
                point_labels = np.array([1])                        #Positive label. 1 is foreground so we're literally telling MobileSAM that the object we want to segment is in this point.

                masks, scores, _ = sam_predictor.predict(           #Forced execution.
                    point_coords=point_coords, 
                    point_labels=point_labels,
                    multimask_output=True,
                )

            if len(masks) > 0:                                      #If any masks were generated...
                best_idx = np.argmax(scores)                        #...from the 3 generated options, pick the one with the highest mathematical precision score...
                pred_crop = masks[best_idx].astype(np.uint8)        #...and use it.

                #Morphological editing before the area calculus.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   #Since the AIs often leave noise behind, we apply a 3x3 elliptical kernel.
                pred_crop = cv2.morphologyEx(pred_crop, cv2.MORPH_CLOSE, kernel) #Fills small holes left within the pore mask.
                pred_crop = cv2.morphologyEx(pred_crop, cv2.MORPH_OPEN, kernel)  #Erases small pixels left out of the pore mask.
                best_mask_crop_bool = pred_crop > 0                 #Convert to boolean for the final drawing.
                
                full_sam_mask = np.zeros((h_img, w_img), dtype=bool)    #Create an empty canvas...
                full_sam_mask[cy1:cy2, cx1:cx2] = best_mask_crop_bool   #...and paste the mask generated by SAM on it.
                
                strict_yolo_mask = np.zeros((h_img, w_img), dtype=bool) #Create a mask that only allows pixels within the YOLO box.
                strict_yolo_mask[sy1:sy2, sx1:sx2] = True
                
                clipped_mask = full_sam_mask & strict_yolo_mask     #The final mask only exists where SAM and YOLO intersect.
                
                pore_area_px = np.sum(clipped_mask)          #We count exactly how many pixels are in the mask.
                aperture_ratio = (pore_area_px / stoma_area_px) * 100 #Calculate the actual percentage of the stoma that the pore occupies.

                if aperture_ratio < aperture_th:                           #If the pore uses less than 12.5% of the stoma, it's biologically closed.
                    closed_stoma += 1                               #We add one to the closed stoma counter.
                    color_box = (0, 0, 255)                         #Red (Closed)
                    state_txt = f"Cerrado ({aperture_ratio:.1f}%)"  #Apply a label.
                    output_img[clipped_mask] = (output_img[clipped_mask]*0.5) + (color_mask_closed[:3]*0.5) # Alpha Blending.
                else:                                               #If the pore opening is equal or greater than 12.5%
                    open_stoma += 1                                 #We add one to the open stoma counter.
                    color_box = (0, 255, 0)                         #Green (Open).
                    state_txt = f"Abierto ({aperture_ratio:.1f}%)"  #Label.
                    output_img[clipped_mask] = (output_img[clipped_mask]*0.5) + (color_mask_open[:3]*0.5)   #Blending.
            else:
                closed_stoma += 1
                color_box = (0, 0, 255)
                state_txt = "Cerrado (Fallo SAM)"

            cv2.rectangle(output_img, (sx1, sy1), (sx2, sy2), color_box, 2) #OpenCV draws a rectangle around the stoma.
            cv2.putText(output_img, state_txt, (sx1, sy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_box, 1) # Write the text.

        output_filename = os.path.join(output_dir, f"analyzed_{img_name}") #Create the new image with the segmentation.
        cv2.imwrite(output_filename, output_img)                    #Save it to the directory.
        print(f"[+] Máscaras aplicadas y resultados guardados en: {output_filename}")

        valid_stomas = closed_stoma + open_stoma                    #Number of total stomas.
        if valid_stomas > 0:                                        #If stomas were detected...
            closed_pct = (closed_stoma / valid_stomas) * 100        #...we obtain the percentage of closed stomas in the image.
            
            if closed_pct <= healthy_th:
                veredict = "Healthy Plant (Control)"
            elif closed_pct <= moderate_th: 
                veredict = "Moderate Hydric Stress"
            else: 
                veredict = "Severe Drought"

            print(f"--[x]--[ Results of {img_name} ]--[x]--")
            print(f"Analyzed Stoma: {valid_stomas}")
            print(f"Open: {open_stoma} || Closed: {closed_stoma}")
            print(f"Closed %: {closed_pct:.2f}%")
            print(f"Diagnostic: {veredict}")
        else:
            print("No se detectaron estomas válidos.")

def main():
    parser = argparse.ArgumentParser(description="Phytolearning Package: Stomatic Hydric Stress Detection")
    parser.add_argument("-i", "--input", required=True, help="Path to the image file or image directory.")
    parser.add_argument("-o", "--output", default="./phyto_output", help="Directory to save the results.")
    parser.add_argument("--aperture", type=float, default=12.5, help="Minimum threshold (%%) for the pore to be considered open (default: 12.5)")
    parser.add_argument("--healthy", type=float, default=25.0, help="Maximum threshold (%%) for 'Healthy Plant' (default: 25.0)")
    parser.add_argument("--moderate", type=float, default=75.0, help="Maximum threshold (%%) for 'Moderate Hydric Stress' (default: 75.0)")
    args = parser.parse_args()
    
    if args.healthy >= args.moderate:
        parser.error("The '--healthy' threshold must be lesser than the '--moderate' threshold.")

    analyze_images(args.input, args.output, aperture_th=args.aperture, healthy_th=args.healthy, moderate_th=args.moderate)

if __name__ == "__main__":
    main()