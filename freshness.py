# from flask import Flask, request, jsonify, send_file
# import torch
# from ultralytics import YOLO
# from torchvision import transforms
# from PIL import Image
# import qrcode
# from io import BytesIO
# import torch.nn as nn
# import cv2

# app = Flask(__name__)

# # Load YOLO model
# yolo_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt'
# yolo_model = YOLO(yolo_model_path)

# # Class definitions for AlexNet and GoogLeNet
# class AlexNet(nn.Module):
#     def __init__(self, num_classes=14):
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

# class GoogLeNet(nn.Module):
#     def __init__(self, num_classes=14):
#         super(GoogLeNet, self).__init__()
#         # Define layers and architecture as given previously

#     def forward(self, x):
#         # Define forward propagation as given previously
#         pass

# # Load the models with their paths
# alexnet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth"
# googlenet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth"
# class_names = ["Apple", "Banana", "Cherry", "Grape", "Kiwi", "Mango", "Orange", "Peach", "Pear", "Plum", "Strawberry", "Watermelon"]

# def load_models(alexnet_path, googlenet_path, num_classes=14):
#     # Loading AlexNet
#     alexnet = models.alexnet(pretrained=False)
#     alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)
#     alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")), strict=False)

#     # Loading GoogleNet
#     googlenet = models.googlenet(pretrained=False)
#     googlenet.fc = nn.Linear(googlenet.fc.in_features, num_classes)
#     googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")), strict=False)

#     return alexnet, googlenet


# # Preprocess image for AlexNet and GoogLeNet
# def preprocess_image(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0)
#     return image

# @app.route('/yolo_predict', methods=['POST'])
# def yolo_predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
    
#     image_file = request.files['image']
#     image_path = "temp_image.jpg"
#     image_file.save(image_path)

#     results = yolo_model.predict(source=image_path, conf=0.25, save=True, save_dir="predictions")
#     return jsonify({"message": "YOLO prediction completed", "results": str(results)})

# @app.route('/predict', methods=['POST'])
# def predict_route():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400
    
#     image_file = request.files['image']
#     image_path = "temp_image.jpg"
#     image_file.save(image_path)
    
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

#         googlenet_output = googlenet(image)
#         googlenet_probs = torch.softmax(googlenet_output, dim=1)
#         _, googlenet_pred = torch.max(googlenet_probs, 1)
#         googlenet_class = class_names[googlenet_pred.item()]
#         googlenet_confidence = googlenet_probs[0, googlenet_pred].item()

#     return jsonify({
#         "AlexNet Prediction": {"Class": alexnet_class, "Confidence": alexnet_confidence},
#         "GoogLeNet Prediction": {"Class": googlenet_class, "Confidence": googlenet_confidence}
#     })

# @app.route('/generate_qr_code', methods=['POST'])
# def generate_qr_code_route():
#     data = request.json
#     if 'fruit_info' not in data:
#         return jsonify({"error": "No fruit_info provided"}), 400
    
#     fruit_info = data['fruit_info']
#     qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
#     qr.add_data(fruit_info)
#     qr.make(fit=True)
#     img = qr.make_image(fill_color="black", back_color="white")
    
#     img_io = BytesIO()
#     img.save(img_io, 'PNG')
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/png')

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, request, jsonify
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)

# Load YOLO model
# yolo_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/best.pt'
yolo_model_path = 'C:/Users/KSHIT/Downloads/best.pt'
yolo_model = YOLO(yolo_model_path)

# Class definitions for AlexNet and GoogLeNet
class AlexNet(nn.Module):
    def __init__(self, num_classes=14):
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

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=14):
        super(GoogLeNet, self).__init__()
        # Define layers and architecture as given previously

    def forward(self, x):
        # Define forward propagation as given previously
        pass

# Load the models with their paths
alexnet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth"
googlenet_path = "C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth"
class_names = ["Apple", "Banana", "Cherry", "Grape", "Kiwi", "Mango", "Orange", "Peach", "Pear", "Plum", "Strawberry", "Watermelon"]

def load_models(alexnet_path, googlenet_path, num_classes=14):
    # Loading AlexNet
    alexnet = models.alexnet(pretrained=False)
    alexnet.classifier[6] = nn.Linear(alexnet.classifier[6].in_features, num_classes)
    alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")), strict=False)

    # Loading GoogleNet
    googlenet = models.googlenet(pretrained=False)
    googlenet.fc = nn.Linear(googlenet.fc.in_features, num_classes)
    googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")), strict=False)

    return alexnet, googlenet


# Preprocess image for AlexNet and GoogLeNet
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

@app.route('/yolo_predict', methods=['POST'])
def yolo_predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = "temp_image.jpg"
    image_file.save(image_path)

    results = yolo_model.predict(source=image_path, conf=0.25, save=True, save_dir="predictions")
    return jsonify({"message": "YOLO prediction completed", "results": str(results)})

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    image_file = request.files['image']
    image_path = "temp_image.jpg"
    image_file.save(image_path)
    
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

        googlenet_output = googlenet(image)
        googlenet_probs = torch.softmax(googlenet_output, dim=1)
        _, googlenet_pred = torch.max(googlenet_probs, 1)
        googlenet_class = class_names[googlenet_pred.item()]
        googlenet_confidence = googlenet_probs[0, googlenet_pred].item()

    return jsonify({
        "AlexNet Prediction": {"Class": alexnet_class, "Confidence": alexnet_confidence},
        "GoogLeNet Prediction": {"Class": googlenet_class, "Confidence": googlenet_confidence}
    })

if __name__ == '__main__':
    app.run(debug=True)
