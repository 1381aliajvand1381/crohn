import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50
from PIL import Image
import io
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import requests
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ============ تنظیمات ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Server starting on {DEVICE}")

# ============ Groq API - نهایی و فعال ============
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# کلاس‌های تشخیص
CLASS_NAMES = ['normal', 'crohn', 'ulcerative-colitis']
CLASS_NAMES_FA = {
    'normal': 'نرمال',
    'crohn': 'کرون',
    'ulcerative-colitis': 'کولیت اولسراتیو'
}

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

# ============ بارگذاری مدل ============
model = None
model_loaded = False

try:
    model_path = 'models/final_ibd_model.pth'
    if os.path.exists(model_path):
        print(f"📂 فایل مدل پیدا شد: {model_path}")
        model = IBDResNet(num_classes=3)
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        model_loaded = True
        
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ مدل ResNet50 با موفقیت لود شد")
        print(f"   - Device: {DEVICE}")
        print(f"   - حجم فایل: {model_size:.1f} MB")
    else:
        print(f"❌ فایل مدل یافت نشد: {model_path}")
        
except Exception as e:
    print(f"❌ خطا در لود مدل: {e}")
    import traceback
    traceback.print_exc()

# ============ پیش‌پردازش تصویر ============
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ تابع جمله‌بندی با Groq ============
def format_with_groq(disease_fa, confidence):
    """جمله‌بندی با Groq Llama 3.3 (رایگان - فعال)"""
    
    prompts = {
        'normal': f"تشخیص: {disease_fa} با اطمینان {confidence:.1f}%. یک جمله ساده و دوستانه که به کاربر اطمینان بدهد و بگوید نگران نباشد بنویس.",
        'crohn': f"تشخیص: {disease_fa} با اطمینان {confidence:.1f}%. یک جمله ساده و دلسوزانه که کاربر را به مشورت با پزشک متخصص تشویق کند بنویس.",
        'ulcerative-colitis': f"تشخیص: {disease_fa} با اطمینان {confidence:.1f}%. یک جمله ساده و دلسوزانه که کاربر را به پیگیری درمان و مشورت با پزشک تشویق کند بنویس."
    }
    
    if 'نرمال' in disease_fa:
        prompt_key = 'normal'
    elif 'کرون' in disease_fa:
        prompt_key = 'crohn'
    else:
        prompt_key = 'ulcerative-colitis'
    
    prompt = prompts.get(prompt_key, f"تشخیص: {disease_fa} با اطمینان {confidence:.1f}%. یک جمله ساده و دوستانه بنویس.")
    
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "تو یک دستیار پزشکی مهربان هستی. پاسخ‌ها را به زبان فارسی، کوتاه و مفید بده. از کلمات تخصصی سنگین استفاده نکن."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 80,
                "temperature": 0.4
            },
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()
            reply = reply.strip('"').strip("'").strip()
            print(f"🟡 Groq 70B پاسخ: {reply}")
            return reply
        else:
            print(f"⚠️ Groq 70B خطا: {response.status_code}")
            return fallback_groq_8b(disease_fa, confidence)
            
    except Exception as e:
        print(f"⚠️ Groq 70B خطا: {e}")
        return fallback_groq_8b(disease_fa, confidence)

def fallback_groq_8b(disease_fa, confidence):
    """مدل پشتیبان - سریعتر"""
    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "پاسخ فارسی کوتاه."},
                    {"role": "user", "content": f"تشخیص: {disease_fa} با اطمینان {confidence:.1f}%. یک جمله کوتاه بنویس."}
                ],
                "max_tokens": 60,
                "temperature": 0.3
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            reply = result["choices"][0]["message"]["content"].strip()
            print(f"🟢 Groq 8B پاسخ: {reply}")
            return reply
        else:
            return f"✅ تشخیص: {disease_fa} با اطمینان {confidence:.1f}%"
            
    except Exception as e:
        print(f"⚠️ Groq 8B خطا: {e}")
        return f"✅ تشخیص: {disease_fa} با اطمینان {confidence:.1f}%"

# ============ مسیرهای API ============

@app.route('/')
def index():
    """صفحه اصلی چت بات"""
    return render_template('chat.html')

@app.route('/health')
def health():
    """بررسی سلامت سرور"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(DEVICE),
        'timestamp': datetime.now().isoformat(),
        'groq_ready': GROQ_API_KEY is not None
    })

@app.route('/api/test', methods=['GET'])
def test():
    """تست ساده API"""
    return jsonify({
        'success': True,
        'message': 'سرور Crohn IBD Detector فعال است',
        'model_loaded': model_loaded,
        'device': str(DEVICE),
        'llm_configured': GROQ_API_KEY is not None,
        'active_model': 'llama-3.3-70b-versatile',
        'fallback_model': 'llama-3.1-8b-instant'
    })

@app.route('/api/test-groq', methods=['GET'])
def test_groq():
    """تست اتصال به Groq"""
    try:
        test_response = format_with_groq("کرون", 87.5)
        if test_response and not test_response.startswith('✅'):
            return jsonify({
                'success': True,
                'message': 'اتصال به Groq برقرار است',
                'response': test_response,
                'model_used': 'llama-3.3-70b-versatile'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Groq پاسخ نداد',
                'fallback_used': True
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """پیش‌بینی تصویر آندوسکوپی"""
    
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'مدل هنوز لود نشده است. لطفاً چند لحظه دیگر تلاش کنید.'
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'درخواست خالی است'}), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'error': 'عکسی ارسال نشده'}), 400
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': 'فرمت تصویر نامعتبر است'}), 400
        
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        class_idx = prediction.item()
        class_name = CLASS_NAMES[class_idx]
        class_name_fa = CLASS_NAMES_FA[class_name]
        confidence_score = confidence.item()
        
        print(f"✅ پیش‌بینی: {class_name_fa} | اطمینان: {confidence_score:.1%}")
        
        groq_response = format_with_groq(class_name_fa, confidence_score * 100)
        
        if not groq_response:
            if class_name == 'normal':
                groq_response = f"✅ تصویر آندوسکوپی شما نرمال است. با اطمینان {confidence_score*100:.1f}% هیچ نشانه‌ای از التهاب مشاهده نشد."
            elif class_name == 'crohn':
                groq_response = f"⚠️ بر اساس تحلیل تصویر با اطمینان {confidence_score*100:.1f}%، یافته‌ها با بیماری کرون سازگار است. توصیه می‌شود با پزشک متخصص مشورت کنید."
            else:
                groq_response = f"⚠️ بر اساس تحلیل تصویر با اطمینان {confidence_score*100:.1f}%، یافته‌ها با کولیت اولسراتیو سازگار است. لطفاً برای تشخیص قطعی به پزشک مراجعه کنید."
        
        return jsonify({
            'success': True,
            'class': class_name,
            'class_fa': class_name_fa,
            'confidence': float(confidence_score),
            'confidence_percent': f"{confidence_score*100:.1f}%",
            'explanation': groq_response,
            'groq_used': groq_response is not None and not groq_response.startswith('✅') and not groq_response.startswith('⚠️'),
            'model': 'ResNet50 + Groq Llama 3.3'
        })
        
    except Exception as e:
        print(f"❌ خطا در پردازش: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'خطای داخلی سرور: {str(e)}'
        }), 500

# ============ اجرای برنامه ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*60)
    print("🚀 Crohn IBD Detector Server - نسخه نهایی")
    print("="*60)
    print(f"🌐 Server running on port {port}")
    print(f"🧠 مدل ResNet50: {'✅ لود شد' if model_loaded else '❌ لود نشد'}")
    print(f"🦙 Groq API: ✅ فعال (llama-3.3-70b-versatile)")
    print(f"⚡ Fallback: ✅ فعال (llama-3.1-8b-instant)")
    print(f"📡 Endpoints:")
    print(f"   - GET  /")
    print(f"   - GET  /health")
    print(f"   - GET  /api/test")
    print(f"   - GET  /api/test-groq")
    print(f"   - POST /api/predict")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port)
