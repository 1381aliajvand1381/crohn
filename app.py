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

# ============ 1️⃣ مدل ResNet50 خودت ============
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

try:
    checkpoint = torch.load('models/final_ibd_model.pth', map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ مدل ResNet50 لود شد")
except Exception as e:
    print(f"❌ خطا در لود مدل: {e}")

model = model.to(device)
model.eval()

# کلاس‌ها
class_names = ['normal', 'crohn', 'ulcerative-colitis']
class_names_fa = {
    'normal': 'نرمال',
    'crohn': 'کرون',
    'ulcerative-colitis': 'کولیت اولسراتیو'
}

# پیش‌پردازش تصویر
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ 2️⃣ LLM برای جمله‌بندی ============
OPENROUTER_API_KEY = "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"
SITE_URL = "https://crohn-1.onrender.com"
SITE_NAME = "Crohn IBD Detector"

def generate_llm_response(disease_name, confidence, language='fa'):
    """
    ارسال نتیجه مدل به LLM برای جمله‌بندی
    """
    
    # زبان کاربر
    lang_instruction = "به زبان فارسی پاسخ بده." if language == 'fa' else "Answer in English."
    
    # پرامپت هوشمند
    prompt = f"""
    شما یک دستیار پزشکی متخصص در تشخیص بیماری‌های گوارشی هستید.
    
    نتیجه تشخیص مدل هوش مصنوعی:
    - بیماری: {disease_name}
    - درصد اطمینان: {confidence:.1%}
    
    وظیفه شما:
    1. این نتیجه را در قالب یک جمله روان و دوستانه به کاربر توضیح بده
    2. اگر بیماری تشخیص داده شده، به کاربر توصیه کن با پزشک مشورت کند
    3. اگر نرمال است، با آرامش به کاربر اطلاع بده
    4. از کلمات تخصصی سنگین استفاده نکن
    
    {lang_instruction}
    """
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
            },
            json={
                "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
                "messages": [
                    {"role": "system", "content": "تو یک دستیار پزشکی هستی که نتایج تشخیص را به زبان ساده توضیح می‌دهی."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3,
            },
            timeout=10
        )
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            # fallback به جمله پیش‌فرض
            return get_fallback_response(disease_name, confidence)
            
    except Exception as e:
        print(f"⚠️ خطا در ارتباط با LLM: {e}")
        return get_fallback_response(disease_name, confidence)

def get_fallback_response(disease_name, confidence):
    """جملات پیش‌فرض در صورت عدم دسترسی به LLM"""
    confidence_percent = f"{confidence*100:.1f}%"
    
    fallbacks = {
        'normal': f"✅ تصویر آندوسکوپی شما نرمال ارزیابی شد. با اطمینان {confidence_percent} هیچ نشانه‌ای از التهاب یا بیماری مشاهده نشد.",
        'crohn': f"⚠️ بر اساس تحلیل تصویر با دقت {confidence_percent}، یافته‌ها با بیماری کرون سازگار است. توصیه می‌شود برای تشخیص قطعی به پزشک متخصص مراجعه کنید.",
        'ulcerative-colitis': f"⚠️ تصویر شما با احتمال {confidence_percent} علائم کولیت اولسراتیو را نشان می‌دهد. لطفاً برای بررسی بیشتر با پزشک خود مشورت کنید."
    }
    
    return fallbacks.get(disease_name, "نتیجه تشخیص توسط هوش مصنوعی آماده شد.")

# ============ 3️⃣ صفحه اصلی ============
@app.route('/')
def index():
    return render_template('chat.html')

# ============ 4️⃣ پیش‌بینی + LLM ============
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        image_data = data.get('image')
        language = data.get('language', 'fa')  # زبان کاربر
        
        if not image_data:
            return jsonify({'error': 'عکسی ارسال نشده'}), 400
        
        # پردازش تصویر
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # پیش‌بینی با ResNet50
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        class_idx = prediction.item()
        class_name = class_names[class_idx]
        class_name_fa = class_names_fa[class_name]
        confidence_score = confidence.item()
        
        # 🟡 ارسال به LLM برای جمله‌بندی
        llm_response = generate_llm_response(
            disease_name=class_name_fa,
            confidence=confidence_score,
            language=language
        )
        
        return jsonify({
            'class': class_name,
            'class_fa': class_name_fa,
            'confidence': float(confidence_score),
            'confidence_percent': f"{confidence_score*100:.1f}%",
            'explanation': llm_response,  # ✅ پاسخ LLM
            'fallback': False
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
