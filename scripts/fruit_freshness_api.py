from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import os

import torch.nn as nn

app = Flask(__name__)

# Define AlexNet and GoogLeNet classes here (as per your provided code)
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

# Define GoogLeNet model
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



# Load models
def load_models(alexnet_path, googlenet_path, num_classes=14):
    alexnet = AlexNet(num_classes=num_classes)
    googlenet = GoogLeNet(num_classes=num_classes)

    alexnet.load_state_dict(torch.load(alexnet_path, map_location=torch.device("cpu")))
    googlenet.load_state_dict(torch.load(googlenet_path, map_location=torch.device("cpu")))

    alexnet.eval()
    googlenet.eval()

    return alexnet, googlenet

# Preprocess the image for prediction
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image

# Prediction function
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

# Load the models and class names
alexnet_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/alexnet_model.pth'
googlenet_model_path = 'C:/Users/KSHIT/OneDrive/Desktop/FruitFreshnessDetection/models/googlenet_model.pth'
class_names = [
    "rottenapples", "rottenbananas", "rottencucumber", "rottenokra", 
    "rottenoranges", "rottenpotato", "rottentomato", 
    "freshapples", "freshbananas", "freshcucumber", "freshokra", 
    "freshoranges", "freshpotato", "freshtomato"
]
alexnet, googlenet = load_models(alexnet_model_path, googlenet_model_path)

# Define Flask route
@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_path = os.path.join("uploads", image_file.filename)
    image_file.save(image_path)

    try:
        result = predict(image_path, alexnet, googlenet, class_names)
        os.remove(image_path)  # Clean up the uploaded image after prediction
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
