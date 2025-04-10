# import cv2
# import numpy as np
# from pyzbar.pyzbar import decode
# import time

# # Initialize the camera
# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # Set the width
# cam.set(4, 480)  # Set the height

# # Loop to keep the camera on and process frames
# while True:
#     success, frame = cam.read()
#     if not success:
#         break

#     # Decode any QR codes in the frame
#     for barcode in decode(frame):
#         qr_type = barcode.type
#         qr_data = barcode.data.decode('utf-8')
        
#         print(f"Type: {qr_type}")
#         print(f"Data: {qr_data}")
        
#         # Draw a rectangle around the QR code
#         pts = barcode.polygon
#         if len(pts) == 4:
#             pts = [(p.x, p.y) for p in pts]
#             pts = np.array(pts, dtype=np.int32)
#             cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        
#         # Display the decoded information on the frame
#         cv2.putText(frame, qr_data, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # Pause briefly to avoid continuous decoding
#         time.sleep(6)

#     # Display the video feed with detected QR codes
#     cv2.imshow("QR_Scanner", frame)
    
#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cam.release()
# cv2.destroyAllWindows()



import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time

def qr_code_reader(camera_index=0, width=640, height=480, display=True, delay=6):
    """
    Reads and decodes QR codes from the video stream.
    
    Parameters:
    - camera_index (int): The index of the camera (default is 0).
    - width (int): The width of the video capture window.
    - height (int): The height of the video capture window.
    - display (bool): If True, displays the video feed with QR code overlays.
    - delay (int): Time in seconds to pause after reading a QR code.
    """
    # Initialize the camera
    cam = cv2.VideoCapture(camera_index)
    cam.set(3, width)  # Set the width
    cam.set(4, height)  # Set the height

    # Loop to keep the camera on and process frames
    while True:
        success, frame = cam.read()
        if not success:
            break

        # Decode any QR codes in the frame
        for barcode in decode(frame):
            qr_type = barcode.type
            qr_data = barcode.data.decode('utf-8')
            
            print(f"Type: {qr_type}")
            print(f"Data: {qr_data}")
            
            # Draw a rectangle around the QR code
            pts = barcode.polygon
            if len(pts) == 4:
                pts = [(p.x, p.y) for p in pts]
                pts = np.array(pts, dtype=np.int32)
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            
            # Display the decoded information on the frame
            cv2.putText(frame, qr_data, (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Pause briefly to avoid continuous decoding
            time.sleep(delay)

        # Display the video feed with detected QR codes
        if display:
            cv2.imshow("QR Scanner", frame)
        
        # Check for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cam.release()
    cv2.destroyAllWindows()

# Run the QR code reader
qr_code_reader()
