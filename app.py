import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, render_template
import numpy as np
import requests
import json

app = Flask(__name__)

# ============ مدل ResNet50 ============
class IBDResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(IBDResNet, self).__init__()
        self.backbone = resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# بارگذاری مدل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IBDResNet(num_classes=3)

# چک کردن وجود فایل مدل
model_path = 'models/final_ibd_model.pth'
if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✅ مدل ResNet50 لود شد")
    except Exception as e:
        print(f"❌ خطا در لود مدل: {e}")
else:
    print(f"❌ فایل مدل یافت نشد: {model_path}")

model = model.to(device)
model.eval()

# کلاس‌ها
class_names = ['normal', 'crohn', 'ulcerative-colitis']
class_names_fa = {
    'normal': 'نرمال',
    'crohn': 'کرون',
    'ulcerative-colitis': 'کولیت اولسراتیو'
}

# پیش‌پردازش
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ تست سلامت ============
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

# ============ صفحه اصلی ============
@app.route('/')
def index():
    return render_template('chat.html')

# ============ پیش‌بینی ============
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'درخواست خالی است'}), 400
            
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'عکسی ارسال نشده'}), 400
        
        # پردازش تصویر
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # پیش‌بینی
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        class_idx = prediction.item()
        class_name = class_names[class_idx]
        class_name_fa = class_names_fa[class_name]
        confidence_score = confidence.item()
        
        # جمله‌بندی ساده
        if class_name == 'normal':
            explanation = f"✅ تشخیص: نرمال با اطمینان {confidence_score*100:.1f}%"
        elif class_name == 'crohn':
            explanation = f"⚠️ تشخیص: کرون با اطمینان {confidence_score*100:.1f}%"
        else:
            explanation = f"⚠️ تشخیص: کولیت اولسراتیو با اطمینان {confidence_score*100:.1f}%"
        
        return jsonify({
            'success': True,
            'class': class_name,
            'class_fa': class_name_fa,
            'confidence': float(confidence_score),
            'confidence_percent': f"{confidence_score*100:.1f}%",
            'explanation': explanation
        })
        
    except Exception as e:
        print(f"❌ خطا: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
