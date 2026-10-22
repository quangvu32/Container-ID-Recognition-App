import cv2
import numpy as np
from PIL import Image
from imutils import contours

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def text_detection(img_path, text_det_model):
    text_det_results = text_det_model(img_path, verbose=False)[0]
    bboxes = text_det_results.boxes.xyxy.tolist()
    classes = text_det_results.boxes.cls.tolist()
    names = text_det_results.names
    confs = text_det_results.boxes.conf.tolist()
    return bboxes, classes, names, confs

def draw_detections(img, detections):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')

    for bbox in detections:
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(rect)
    
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    
    image_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4) # Reshape to RGBA
    image_array = image_array[:, :, :3] # Remove Alpha, keep only RGB
    
    plt.close(fig)
    return image_array

def predict(img_path, bboxes):
    # Load the image
    img = Image.open(img_path).convert('RGB')

    # Draw and return the annotated image
    annotated_img = draw_detections(img, bboxes)
    return annotated_img

def split_bounding_boxes(img_path, bboxes):
    #os.makedirs(save_dir, exist_ok=True)
    cropped_imgs = []
    count = 0
    img = Image.open(img_path).convert("L")  # Convert to grayscale for np.mean
    for bb in bboxes:
        # If bb is in (x1, y1, x2, y2) format
        cropped_img = img.crop((bb[0], bb[1], bb[2], bb[3]))
        
        # Convert to NumPy for pixel value checking
        cropped_arr = np.array(cropped_img)

        # Filter out nearly black or white
        if np.mean(cropped_arr) < 35 or np.mean(cropped_arr) > 220:
            continue
        if cropped_img.size[0] < 10 or cropped_img.size[1] < 10:
            continue
        cropped_imgs.append(cropped_img)
    
    return cropped_imgs

def create_char_imgs(img_path, bboxes, padding = 5, min_area = 50, min_width = 5, min_height = 20):  
    '''
    padding: adjust the bb expansion
    min_area: filter out too-small contours
    min_width: minimum contour width
    min_height: minimum contour height
    '''
    char_batch = []
    cropped_imgs = split_bounding_boxes(img_path, bboxes)

    for i, img in enumerate(cropped_imgs):
        x1, y1, x2, y2 = bboxes[i] # bb of each cropped image 
        image = np.array(img)
        # Threshold
        thresh = cv2.threshold(image,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1] 
        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts[0] if len(cnts) == 2 else cnts[1] 
        cnts, _ = contours.sort_contours(cnts, method="top-to-bottom") if y2 - y1 > x2 - x1 else contours.sort_contours(cnts, method="left-to-right") 

        char_imgs = []
        ROI_number = 0
        for c in cnts:
            area = cv2.contourArea(c) 
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            # Filtering: remove tiny specks and extremely thin/wide shapes
            if area > min_area and w > min_width and h > min_height and 0.2 < aspect_ratio < 5:
                # Expand bounding box
                x_new = max(x - padding, 0)
                y_new = max(y - padding, 0)
                w_new = min(w + 2*padding, image.shape[1] - x_new)
                h_new = min(h + 2*padding, image.shape[0] - y_new)

                ROI = image[y_new:y_new+h_new, x_new:x_new+w_new] 
                #ROI = image[y:y+h, x:x+w] 
                char_imgs.append(ROI) 

        char_batch.append(char_imgs)
    
    return char_batch

def create_horizontal_strip(char_imgs, spacing=5):
    if len(char_imgs) == 0:
        return None
    # Fixed height for all characters
    target_height = max(c.shape[0] for c in char_imgs)
    resized_chars = [
        cv2.resize(c, (int(c.shape[1] * target_height / c.shape[0]), target_height))
        for c in char_imgs
    ]
    # Get background color from the first image (can be changed)
    background_color = int(np.mean([
        np.mean(c[:, 0]) for c in resized_chars  # Use the first channel for background
    ]))
    strip_parts = []
    for c in resized_chars:
        strip_parts.append(c)
        #background_color = int(np.mean([np.mean(c[:, 0])]))
        strip_parts.append(
            np.full((target_height, spacing), background_color, dtype=np.uint8)
        )
    horizontal_strip = np.hstack(strip_parts[:-1])  # Remove last spacing
    return horizontal_strip


