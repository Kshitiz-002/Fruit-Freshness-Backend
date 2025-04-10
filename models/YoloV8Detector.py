# import cv2
# from ultralytics import YOLO

# # Load the YOLO model with the weights file
# # model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/Dev/runs/detect/train7/weights/best.pt')
# model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt')

# # Define class names
# class_names = ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Perform detection
#     results = model(frame)
    
#     # Iterate over each detection in results
#     for result in results:
#         if result.boxes is not None:  # Check if there are any detections
#             for box in result.boxes:
#                 # Extract bounding box coordinates, confidence, and class id
#                 xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
#                 confidence = box.conf[0].item()
#                 class_id = int(box.cls[0].item())
                
#                 # Get label
#                 label = f"{class_names[class_id]}: {confidence:.2f}"
                
#                 # Draw bounding box and label
#                 cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
#                 cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
#     # Display the frame
#     cv2.imshow('YOLO Real-Time Detection', frame)
    
#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()







# import cv2
# from ultralytics import YOLO

# # Load the YOLO model with the weights file
# model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt')

# # Define class names
# class_names = ['Apple', 'Banana', 'Grape', 'Orange', 'Pineapple', 'Watermelon']

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Perform detection
#     results = model(frame)
    
#     # Iterate over each detection in results
#     for result in results:
#         if result.boxes is not None:  # Check if there are any detections
#             for box in result.boxes:
#                 # Extract bounding box coordinates, confidence, and class id
#                 xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
#                 confidence = box.conf[0].item()
#                 class_id = int(box.cls[0].item())
                
#                 # Draw bounding box and label only if confidence > 80%
#                 if confidence >= 0.5:
#                     label = f"{class_names[class_id]}: {confidence * 100:.2f}%"  # Confidence as a percentage
#                     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
#                     cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
#     # Display the frame
#     cv2.imshow('YOLO Real-Time Detection', frame)
    
#     # Break loop with 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO

def detect_objects_in_real_time(model_path, class_names, confidence_threshold=0.8):
    """
    Detects objects in real-time using a YOLO model and displays the results with bounding boxes.
    
    Parameters:
    - model_path (str): Path to the YOLO model weights file.
    - class_names (list): List of class names corresponding to the model's output classes.
    - confidence_threshold (float): Minimum confidence level to display a bounding box (default is 0.8).
    """
    # Load the YOLO model with the provided weights file
    model = YOLO(model_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)
        
        # Iterate over each detection in results
        for result in results:
            if result.boxes is not None:  # Check if there are any detections
                for box in result.boxes:
                    # Extract bounding box coordinates, confidence, and class id
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    # Draw bounding box and label only if confidence > confidence_threshold
                    if confidence >= confidence_threshold:
                        label = f"{class_names[class_id]}: {confidence * 100:.2f}%"  # Confidence as a percentage
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                        cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display the frame
        cv2.imshow('YOLO Real-Time Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
