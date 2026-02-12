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
CORS(app)  # برای رفع مشکل CORS

# ============ تنظیمات ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Server starting on {DEVICE}")

# ✅ API Key OpenRouter - رایگان
OPENROUTER_API_KEY = "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"

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
        
        # بارگذاری مدل
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        model_loaded = True
        
        # محاسبه حجم مدل
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ مدل ResNet50 با موفقیت لود شد")
        print(f"   - Device: {DEVICE}")
        print(f"   - حجم فایل: {model_size:.1f} MB")
        print(f"   - کلاس‌ها: {CLASS_NAMES}")
    else:
        print(f"❌ فایل مدل یافت نشد: {model_path}")
        print(f"   - مسیر جاری: {os.getcwd()}")
        print(f"   - محتویات پوشه models: {os.listdir('models') if os.path.exists('models') else 'پوشه وجود ندارد'}")
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

# ============ تابع جمله‌بندی با OpenRouter (رایگان) ============
def format_with_llm(disease_fa, confidence):
    """
    تبدیل نتیجه تشخیص به جمله روان با Llama 3.2 Vision
    مدل: meta-llama/llama-3.2-11b-vision-instruct:free (کاملاً رایگان)
    """
    
    # پرامپت ساده و مستقیم
    prompt = f"""به عنوان یک دستیار پزشکی، این نتیجه تشخیص را به یک جمله ساده و دوستانه تبدیل کن:

تشخیص: {disease_fa}
درصد اطمینان: {confidence:.1f}%

فقط یک جمله کوتاه و روان بنویس، بدون توضیح اضافه."""

    try:
        print(f"🟡 ارسال درخواست به OpenRouter...")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://crohn-1.onrender.com",
                "X-Title": "Crohn IBD Detector"
            },
            json={
                "model": "meta-llama/llama-3.2-11b-vision-instruct:free",  # ✅ مدل رایگان
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.3
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result["choices"][0]["message"]["content"].strip()
            print(f"✅ پاسخ OpenRouter: {llm_response}")
            return llm_response
        else:
            print(f"⚠️ OpenRouter خطا: {response.status_code}")
            print(f"   پاسخ: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"⚠️ OpenRouter خطا: {e}")
        return None

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
        'llm_ready': OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY.startswith('sk-or-')
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """پیش‌بینی تصویر آندوسکوپی"""
    
    # چک کردن مدل
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'مدل هنوز لود نشده است. لطفاً چند لحظه دیگر تلاش کنید.'
        }), 503
    
    try:
        # دریافت داده
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'درخواست خالی است'}), 400
        
        image_data = data.get('image')
        if not image_data:
            return jsonify({'success': False, 'error': 'عکسی ارسال نشده'}), 400
        
        # پردازش تصویر
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': 'فرمت تصویر نامعتبر است'}), 400
        
        # پیش‌بینی با ResNet50
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
        
        # ============ جمله‌بندی با OpenRouter ============
        llm_response = format_with_llm(class_name_fa, confidence_score * 100)
        
        # اگر OpenRouter جواب نداد، از جمله پیش‌فرض استفاده کن
        if not llm_response:
            if class_name == 'normal':
                llm_response = f"✅ تصویر آندوسکوپی شما نرمال است. با اطمینان {confidence_score*100:.1f}% هیچ نشانه‌ای از التهاب مشاهده نشد."
            elif class_name == 'crohn':
                llm_response = f"⚠️ یافته‌های تصویر با بیماری کرون سازگار است. (درصد اطمینان: {confidence_score*100:.1f}%)"
            else:
                llm_response = f"⚠️ یافته‌های تصویر با کولیت اولسراتیو سازگار است. (درصد اطمینان: {confidence_score*100:.1f}%)"
            
            print(f"⚪ استفاده از جمله پیش‌فرض")
        
        # برگرداندن نتیجه
        return jsonify({
            'success': True,
            'class': class_name,
            'class_fa': class_name_fa,
            'confidence': float(confidence_score),
            'confidence_percent': f"{confidence_score*100:.1f}%",
            'explanation': llm_response,
            'llm_used': llm_response is not None and not llm_response.startswith('✅') and not llm_response.startswith('⚠️')
        })
        
    except Exception as e:
        print(f"❌ خطا در پردازش: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'خطای داخلی سرور: {str(e)}'
        }), 500

@app.route('/api/test', methods=['GET'])
def test():
    """تست ساده API"""
    return jsonify({
        'success': True,
        'message': 'سرور Crohn IBD Detector فعال است',
        'model_loaded': model_loaded,
        'device': str(DEVICE),
        'llm_configured': OPENROUTER_API_KEY is not None and OPENROUTER_API_KEY.startswith('sk-or-')
    })

@app.route('/api/test-llm', methods=['GET'])
def test_llm():
    """تست اتصال به OpenRouter"""
    try:
        test_response = format_with_llm("کرون", 85.5)
        if test_response:
            return jsonify({
                'success': True,
                'message': 'اتصال به OpenRouter برقرار است',
                'response': test_response
            })
        else:
            return jsonify({
                'success': False,
                'message': 'OpenRouter پاسخ نداد'
            }), 503
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============ اجرای برنامه ============
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Server running on port {port}")
    print(f"📝 OpenRouter API Key: {'✅ تنظیم شده' if OPENROUTER_API_KEY.startswith('sk-or-') else '❌ تنظیم نشده'}")
    print(f"🧠 مدل: {'✅ لود شد' if model_loaded else '❌ لود نشد'}")
    print("="*50)
    app.run(host='0.0.0.0', port=port)
