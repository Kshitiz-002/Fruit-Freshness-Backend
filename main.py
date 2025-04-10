# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch
# from torchvision import transforms
# from PIL import Image
# import qrcode
# import time
# import torch.nn as nn

# # Load YOLO model
# model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt')



# class AlexNet(nn.Module):
#     def __init__(self, num_classes=14):  # Set num_classes to 14 for your custom dataset
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x

# # Define GoogLeNet model
# class Inception(nn.Module):
#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super(Inception, self).__init__()
#         self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
#             nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
#             nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
#         )
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels, pool_proj, kernel_size=1)
#         )

#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)

# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=14):
#         super(GoogLeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#         x = self.inception4a(x)
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x


# # Load classification models (AlexNet and GoogLeNet)
# def load_models(alexnet_path, googlenet_path, num_classes=14):
#     # Initialize model architectures
#     alexnet = AlexNet(num_classes=num_classes)
#     googlenet = GoogLeNet(num_classes=num_classes)

#     # Load the saved state dictionaries
#     alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")))
#     googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")))

#     alexnet.eval()
#     googlenet.eval()
    
#     return alexnet, googlenet

# # Preprocess the image before classification
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize to model input size
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image)
#     image = image.unsqueeze(0)  # Add batch dimension
#     return image

# # Classify the fruit using AlexNet and GoogLeNet
# def predict(image_path, alexnet, googlenet, class_names):
#     image = preprocess_image(image_path)
    
#     # Device configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     alexnet.to(device)
#     googlenet.to(device)
#     image = image.to(device)

#     # Prediction with AlexNet
#     with torch.no_grad():
#         alexnet_output = alexnet(image)
#         alexnet_probs = torch.softmax(alexnet_output, dim=1)
#         _, alexnet_pred = torch.max(alexnet_probs, 1)
#         alexnet_class = class_names[alexnet_pred.item()]
#         alexnet_confidence = alexnet_probs[0, alexnet_pred].item()

#     # Prediction with GoogLeNet
#     with torch.no_grad():
#         googlenet_output = googlenet(image)
#         googlenet_probs = torch.softmax(googlenet_output, dim=1)
#         _, googlenet_pred = torch.max(googlenet_probs, 1)
#         googlenet_class = class_names[googlenet_pred.item()]
#         googlenet_confidence = googlenet_probs[0, googlenet_pred].item()

#     return {
#         "AlexNet Prediction": {
#             "Class": alexnet_class,
#             "Confidence": alexnet_confidence
#         },
#         "GoogLeNet Prediction": {
#             "Class": googlenet_class,
#             "Confidence": googlenet_confidence
#         }
#     }

# # QR Code Generation function
# def generate_qr_code(fruit_info, output_path="fruit_info_qr.png"):
#     qr = qrcode.QRCode(version=1,
#                        error_correction=qrcode.constants.ERROR_CORRECT_L,
#                        box_size=10,
#                        border=4)
#     qr.add_data(fruit_info)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
#     img.save(output_path)

# # Main function to detect, classify, and generate QR code
# def detect_and_classify_fruit(camera_index=0):
#     # Initialize YOLO model and camera
#     cap = cv2.VideoCapture(camera_index)
#     cap.set(3, 640)  # Set the width
#     cap.set(4, 480)  # Set the height

#     # Load AlexNet and GoogLeNet models
#     alexnet_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth'
#     googlenet_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth'
#     class_names = [
#         "rottenapples", "rottenbananas", "rottencucumber", "rottenokra", 
#         "rottenoranges", "rottenpotato", "rottentomato", 
#         "freshapples", "freshbananas", "freshcucumber", "freshokra", 
#         "freshoranges", "freshpotato", "freshtomato"
#     ]
#     alexnet, googlenet = load_models(alexnet_model_path, googlenet_model_path)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform detection
#         results = model(frame)
        
#         for result in results:
#             if result.boxes is not None:  # If there are detections
#                 for box in result.boxes:
#                     xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
#                     confidence = box.conf[0].item()
#                     class_id = int(box.cls[0].item())
#                     label = f"{class_names[class_id]}: {confidence:.2f}"

#                     # Draw bounding box and label
#                     cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
#                     cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
#                     # Crop the detected region (fruit)
#                     cropped_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
#                     img_path = "captured_fruit.jpg"
#                     cv2.imwrite(img_path, cropped_img)

#                     # Classify the fruit using AlexNet and GoogLeNet
#                     result = predict(img_path, alexnet, googlenet, class_names)
                    
#                     # Prepare fruit information for QR Code
#                     fruit_info = f"AlexNet: {result['AlexNet Prediction']['Class']} ({result['AlexNet Prediction']['Confidence']*100:.2f}%)\n"
#                     fruit_info += f"GoogLeNet: {result['GoogLeNet Prediction']['Class']} ({result['GoogLeNet Prediction']['Confidence']*100:.2f}%)"
                    
#                     # Generate QR Code with the fruit information
#                     generate_qr_code(fruit_info, output_path="fruit_info_qr.png")
#                     print("QR Code generated and saved as fruit_info_qr.png")

#         # Display the frame with detection
#         cv2.imshow("Fruit Detection and QR Code", frame)

#         # Break loop with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the camera and close windows
#     cap.release()
#     cv2.destroyAllWindows()

# # Run the detection and classification
# detect_and_classify_fruit()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch
# from torchvision import transforms
# from PIL import Image
# import qrcode
# import torch.nn as nn

# # Load YOLO model
# yolo_model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt')

# class AlexNet(nn.Module):
#     def __init__(self, num_classes=14):  # Set num_classes to 14 for your custom dataset
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x

# class Inception(nn.Module):
#     def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
#         super(Inception, self).__init__()
#         self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
#             nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
#             nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
#         )
#         self.branch4 = nn.Sequential(
#             nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#             nn.Conv2d(in_channels, pool_proj, kernel_size=1)
#         )

#     def forward(self, x):
#         branch1 = self.branch1(x)
#         branch2 = self.branch2(x)
#         branch3 = self.branch3(x)
#         branch4 = self.branch4(x)
#         outputs = [branch1, branch2, branch3, branch4]
#         return torch.cat(outputs, 1)

# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=14):
#         super(GoogLeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
#         self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
#         self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
#         self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
#         self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
#         self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
#         self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
#         self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
#         self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
#         self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
#         self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.4)
#         self.fc = nn.Linear(1024, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.inception3a(x)
#         x = self.inception3b(x)
#         x = self.maxpool3(x)
#         x = self.inception4a(x)
#         x = self.inception4b(x)
#         x = self.inception4c(x)
#         x = self.inception4d(x)
#         x = self.inception4e(x)
#         x = self.maxpool4(x)
#         x = self.inception5a(x)
#         x = self.inception5b(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return x

# def load_models(alexnet_path, googlenet_path, num_classes=14):
#     alexnet = AlexNet(num_classes=num_classes)
#     googlenet = GoogLeNet(num_classes=num_classes)
#     alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")))
#     googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")))
#     alexnet.eval()
#     googlenet.eval()
#     return alexnet, googlenet

# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)), 
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0)
#     return image

# # Define YOLO prediction and save function
# def yolo_predict(image_path, yolo_model_path='C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt', output_dir='predictions'):
#     # Load the YOLO model
#     model = YOLO(yolo_model_path)
    
#     # Perform prediction
#     results = model.predict(
#         mode='predict', 
#         conf=0.25, 
#         source=image_path, 
#         save=True,  # This enables saving of the prediction output
#         save_dir=output_dir  # Directory to save the output images
#     )
    
#     print(f"Prediction saved in {output_dir}")
#     return results


# def predict(image_path, alexnet, googlenet, class_names):
#     image = preprocess_image(image_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     alexnet.to(device)
#     googlenet.to(device)
#     image = image.to(device)

#     with torch.no_grad():
#         alexnet_output = alexnet(image)
#         alexnet_probs = torch.softmax(alexnet_output, dim=1)
#         _, alexnet_pred = torch.max(alexnet_probs, 1)
#         alexnet_class = class_names[alexnet_pred.item()]
#         alexnet_confidence = alexnet_probs[0, alexnet_pred].item()

#     with torch.no_grad():
#         googlenet_output = googlenet(image)
#         googlenet_probs = torch.softmax(googlenet_output, dim=1)
#         _, googlenet_pred = torch.max(googlenet_probs, 1)
#         googlenet_class = class_names[googlenet_pred.item()]
#         googlenet_confidence = googlenet_probs[0, googlenet_pred].item()

#     return {
#         "AlexNet Prediction": {
#             "Class": alexnet_class,
#             "Confidence": alexnet_confidence
#         },
#         "GoogLeNet Prediction": {
#             "Class": googlenet_class,
#             "Confidence": googlenet_confidence
#         }
#     }

# def generate_qr_code(fruit_info, output_path="fruit_info_qr.png"):
#     qr = qrcode.QRCode(version=1,
#                        error_correction=qrcode.constants.ERROR_CORRECT_L,
#                        box_size=10,
#                        border=4)
#     qr.add_data(fruit_info)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
#     img.save(output_path)

# # Example usage
# alexnet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth"
# googlenet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth"
# class_names = [
#     "Apple", "Banana", "Cherry", "Grape", "Kiwi", "Mango", 
#     "Orange", "Peach", "Pear", "Plum", "Strawberry", "Watermelon"
# ]
#   # replace with actual class names
# alexnet, googlenet = load_models(alexnet_path, googlenet_path, num_classes=14)

# image_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/download.jpeg"
# yolo_results = yolo_predict(image_path)
# print(yolo_results)
# fruit_info = predict(image_path, alexnet, googlenet, class_names)
# print(fruit_info)
# generate_qr_code(fruit_info, output_path="fruit_info_qr.png")


import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Load YOLO model
# yolo_model = YOLO('C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt')
yolo_model_path = 'C:/Users/KSHIT/Downloads/best.pt'

class AlexNet(nn.Module):
    def __init__(self, num_classes=14):  # Set num_classes to 14 for your custom dataset
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=14):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def load_models(alexnet_path, googlenet_path, num_classes=14):
    alexnet = AlexNet(num_classes=num_classes)
    googlenet = GoogLeNet(num_classes=num_classes)
    alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")))
    googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")))
    alexnet.eval()
    googlenet.eval()
    return alexnet, googlenet

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

# Define YOLO prediction and save function
def yolo_predict(image_path, yolo_model_path='C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt', output_dir='predictions'):
    # Load the YOLO model
    model = YOLO(yolo_model_path)
    
    # Perform prediction
    results = model.predict(
        mode='predict', 
        conf=0.25, 
        source=image_path, 
        save=True,  # This enables saving of the prediction output
        save_dir=output_dir  # Directory to save the output images
    )
    
    print(f"Prediction saved in {output_dir}")
    return results


def predict(image_path, alexnet, googlenet, class_names):
    image = preprocess_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alexnet.to(device)
    googlenet.to(device)
    image = image.to(device)

    with torch.no_grad():
        alexnet_output = alexnet(image)
        alexnet_probs = torch.softmax(alexnet_output, dim=1)
        _, alexnet_pred = torch.max(alexnet_probs, 1)
        alexnet_class = class_names[alexnet_pred.item()]
        alexnet_confidence = alexnet_probs[0, alexnet_pred].item()

    with torch.no_grad():
        googlenet_output = googlenet(image)
        googlenet_probs = torch.softmax(googlenet_output, dim=1)
        _, googlenet_pred = torch.max(googlenet_probs, 1)
        googlenet_class = class_names[googlenet_pred.item()]
        googlenet_confidence = googlenet_probs[0, googlenet_pred].item()

    return {
        "AlexNet Prediction": {
            "Class": alexnet_class,
            "Confidence": alexnet_confidence
        },
        "GoogLeNet Prediction": {
            "Class": googlenet_class,
            "Confidence": googlenet_confidence
        }
    }

# Example usage
alexnet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth"
googlenet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth"

class_names = ['freshapples', 'freshbananas', 'freshcucumber', 'freshokra', 'freshoranges', 'freshpotato',
               'freshtomato', 'rottenapples', 'rottenbananas', 'rottencucumber', 'rottenokra', 'rottenoranges', 
               'rottenpotato', 'rottentomato']

alexnet, googlenet = load_models(alexnet_path, googlenet_path)

image_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/download.jpeg"
result = predict(image_path, alexnet, googlenet, class_names)
print(result)
