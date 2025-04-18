# !pip install ultralytics (Don't forget install it)

# For run thhis porgram you must install the next libraries in your machine for.

#--------------------------------------------
# IF DOESN'T WORK JUST DO IT IN A IPYNB FILE
#--------------------------------------------

import cv2
import time
import numpy as np
from ultralytics import YOLO


# Load the model YOLOv11 for segmentation (or have download 'yolo11n-seg.pt')
# You can check https://docs.ultralytics.com/es/tasks/segment/ for other models

model = YOLO("yolo11n-seg")

# Define the source of video (Can be the path o index of cam )
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # To measure time of processing to calculate the latency 
    start_time = time.time()
    
    # Make the detecation and detection on the frame 
    results = model(
        frame, 
        conf=0.7, # Confidence of show over 70 %
        classes=[0] # "0" for focuss just in person
        )
    latency = (time.time() - start_time) * 1000  # miliseconds in latency

    # Acces to adress (bounding boxes)
    boxes_obj = results[0].boxes
    if boxes_obj is not None and len(boxes_obj) > 0:
        bboxes = boxes_obj.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
        confs = boxes_obj.conf.cpu().numpy()      # Confidence scores (Puntajes de confianza)
        classes = boxes_obj.cls.cpu().numpy()     # Class index (Índices de clase)
        
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            
            # Get the class name if this exist 
            class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))
            label = f'{class_name} {confs[i]:.2f}'
            # Draw bounding box and label the frame 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # PProcess the segmetations: assing random color for every mask detected
    masks_obj = results[0].masks
    if masks_obj is not None and len(masks_obj) > 0:
        # Extract the mask: It's assume that masks_obj.data is a tensor
        masks = masks_obj.data.cpu().numpy() if hasattr(masks_obj.data, 'cpu') else masks_obj.data
        for mask in masks:
            # Become the mask to binary (umbral 0.5) and scale to 0-255
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            # Reshape the mask for get the same size in frame 
            mask_bin = cv2.resize(mask_bin, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Create a boolean mask with 3 channels 
            binary_mask = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)[1]
            binary_mask_3c = cv2.merge([binary_mask, binary_mask, binary_mask])
            
            # Generate radom color  (BGR)
            random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            # Create an image the same size as the frame, filled with the random color
            colored_mask = np.full((frame.shape[0], frame.shape[1], 3), random_color, dtype=np.uint8)
            
            # Combine mask with frame: In regions where the mask is 255, the random color is used
            output_frame = frame.copy()
            output_frame[binary_mask_3c == 255] = colored_mask[binary_mask_3c == 255]
            
            # Update the frame with the colored mask (keeping the natural background)
            frame = output_frame
        
        # Display the number of masks detected
        cv2.putText(frame, f'Masks: {len(masks)}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display latency in the frame
    cv2.putText(frame, f'Latency: {latency:.1f}ms', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the processed frame in real time
    cv2.imshow("YOLOv11-Seg - Real Time segementation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()