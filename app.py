from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import torchvision.models as models

app = Flask(__name__)

# Modelinizi yükleyin
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 122)  # class_names.txt dosyasındaki sınıf sayısı kadar
model.load_state_dict(torch.load('trained_model26.pth'))
model.eval()

# Sınıf isimlerini ve besin bilgilerini yükle
food_info = {}
class_names = []
with open('class_names.txt', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        class_name = parts[0]
        class_names.append(class_name)
        food_info[class_name] = {
            "calories": float(parts[1]),
            "carbs": float(parts[2]),
            "protein": float(parts[3]),
            "fat": float(parts[4])
        }

# Görüntü dönüşüm fonksiyonu
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# Tahmin fonksiyonu
def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor)
    _, predicted = outputs.max(1)
    class_name = class_names[predicted.item()]
    return class_name

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({class_name: food_info[class_name]})

if __name__ == '__main__':
    app.run()
