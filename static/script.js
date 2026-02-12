// static/script.js

// ============ تنظیمات ============
const SERVER_URL = 'https://crohn-1.onrender.com';
let currentImage = null;

// ============ المنت‌ها ============
const chatBox = document.getElementById('chatBox');
const loading = document.getElementById('loading');
const sendBtn = document.getElementById('sendBtn');
const messageInput = document.getElementById('messageInput');
const imageInput = document.getElementById('imageInput');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const serverUrl = document.getElementById('serverUrl');

// ============ بررسی وضعیت سرور ============
async function checkServerStatus() {
    try {
        const response = await fetch(`${SERVER_URL}/health`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            mode: 'cors',
            cache: 'no-cache'
        });

        if (response.ok) {
            const data = await response.json();
            
            // به‌روزرسانی وضعیت
            statusDot.className = 'status-dot green';
            statusText.innerHTML = `✅ سرور فعال | مدل: ${data.model_loaded ? '✅' : '❌'} | LLM: ${data.llm_ready ? '✅' : '❌'}`;
            serverUrl.textContent = 'crohn-1.onrender.com';
            
            // فعال کردن دکمه‌ها
            sendBtn.disabled = false;
            messageInput.disabled = false;
            
            // اگه مدل لود نشده
            if (!data.model_loaded) {
                addSystemMessage('⏳ مدل در حال بارگذاری است، لطفاً ۱ دقیقه صبر کنید...');
            }
            
            return true;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('❌ Server connection error:', error);
        
        statusDot.className = 'status-dot red';
        statusText.innerHTML = '❌ قطع ارتباط با سرور';
        serverUrl.textContent = 'عدم اتصال';
        
        sendBtn.disabled = true;
        messageInput.disabled = true;
        
        return false;
    }
}

// ============ اضافه کردن پیام سیستمی ============
function addSystemMessage(text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const timestamp = new Date().toLocaleTimeString('fa-IR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-content" style="background: #ebf8ff; border-color: #4299e1;">
            ${text}
            <div style="margin-top: 8px; font-size: 11px; color: #718096;">🖥️ سیستم</div>
        </div>
        <div class="timestamp">${timestamp}</div>
    `;
    
    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

// ============ پیش‌نمایش عکس ============
imageInput.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // بررسی حجم
        if (file.size > 10 * 1024 * 1024) {
            alert('❌ حجم فایل نباید بیشتر از ۱۰ مگابایت باشد');
            return;
        }
        
        // بررسی فرمت
        if (!file.type.startsWith('image/')) {
            alert('❌ لطفاً فقط فایل تصویری آپلود کنید');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            currentImage = e.target.result;
            
            const previewContainer = document.getElementById('imagePreview');
            previewContainer.innerHTML = `
                <div class="image-preview-wrapper">
                    <img src="${e.target.result}" class="image-preview" alt="پیش‌نمایش">
                    <button onclick="removeImage()" class="remove-btn" title="حذف عکس">✕</button>
                    <span style="position: absolute; bottom: -10px; left: 10px; background: #4299e1; color: white; padding: 5px 12px; border-radius: 20px; font-size: 11px; font-weight: 600;">
                        🖼️ آماده ارسال
                    </span>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }
});

// ============ حذف عکس ============
window.removeImage = function() {
    currentImage = null;
    document.getElementById('imagePreview').innerHTML = '';
    imageInput.value = '';
};

// ============ ارسال به سرور ============
window.sendMessage = async function() {
    // بررسی عکس
    if (!currentImage) {
        addSystemMessage('❌ لطفاً ابتدا یک عکس آندوسکوپی آپلود کنید');
        return;
    }

    // نمایش لودینگ
    loading.style.display = 'block';
    sendBtn.disabled = true;
    messageInput.disabled = true;

    try {
        const response = await fetch(`${SERVER_URL}/api/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage
            })
        });

        const data = await response.json();

        // مخفی کردن لودینگ
        loading.style.display = 'none';
        sendBtn.disabled = false;
        messageInput.disabled = false;

        if (data.success) {
            showResult(data);
        } else {
            addSystemMessage(`❌ خطا: ${data.error || 'خطای ناشناخته'}`);
        }

    } catch (error) {
        console.error('❌ Error:', error);
        
        loading.style.display = 'none';
        sendBtn.disabled = false;
        messageInput.disabled = false;
        
        addSystemMessage(`❌ خطا در ارتباط با سرور: ${error.message}`);
    }
};

// ============ نمایش نتیجه ============
function showResult(data) {
    // انتخاب رنگ بر اساس کلاس
    let color, icon;
    switch (data.class) {
        case 'normal':
            color = '#48bb78';
            icon = '✅';
            break;
        case 'crohn':
            color = '#ed8936';
            icon = '⚠️';
            break;
        case 'ulcerative-colitis':
            color = '#f56565';
            icon = '🔴';
            break;
        default:
            color = '#4299e1';
            icon = 'ℹ️';
    }

    // پیام کاربر (عکس)
    const userMsg = document.createElement('div');
    userMsg.className = 'message user-message';
    userMsg.innerHTML = `
        <div class="message-content" style="max-width: 300px; padding: 10px;">
            <img src="${currentImage}" style="width: 100%; border-radius: 10px;">
            <div style="margin-top: 5px; font-size: 11px; color: rgba(255,255,255,0.8); text-align: center;">
                🖼️ تصویر آندوسکوپی
            </div>
        </div>
        <div class="timestamp">${new Date().toLocaleTimeString('fa-IR')}</div>
    `;
    chatBox.appendChild(userMsg);

    // پاسخ بات
    const botMsg = document.createElement('div');
    botMsg.className = 'message bot-message';
    botMsg.innerHTML = `
        <div class="message-content">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
                <span style="font-size: 32px;">${icon}</span>
                <div>
                    <div style="font-size: 24px; font-weight: 700; color: ${color};">
                        ${data.class_fa}
                    </div>
                    <div style="font-size: 12px; color: #718096;">
                        کد تشخیص: ${data.class}
                    </div>
                </div>
            </div>
            
            <div style="background: #f8fafc; padding: 15px; border-radius: 12px; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="color: #4a5568; font-weight: 600;">درصد اطمینان</span>
                    <span style="font-weight: 700; color: ${color}; font-size: 18px;">
                        ${data.confidence_percent}
                    </span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${data.confidence_percent}; background: ${color};"></div>
                </div>
            </div>
            
            <div style="background: #ebf8ff; padding: 15px; border-radius: 12px; border-right: 4px solid #4299e1;">
                <div style="display: flex; gap: 10px;">
                    <span style="font-size: 20px;">💬</span>
                    <div>
                        <span style="font-weight: 700; color: #2c5282; display: block; margin-bottom: 5px;">
                            پاسخ Llama 3.2:
                        </span>
                        <p style="color: #2d3748; line-height: 1.6; margin: 0;">
                            ${data.explanation}
                        </p>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 15px; display: flex; gap: 10px; justify-content: flex-end;">
                <span class="llm-badge">
                    ${data.llm_used ? '🟡 Llama 3.2' : '⚪ جمله پیش‌فرض'}
                </span>
                <span style="padding: 4px 12px; background: #e2e8f0; border-radius: 20px; font-size: 11px; color: #4a5568;">
                    🧠 ResNet50
                </span>
            </div>
        </div>
        <div class="timestamp">${new Date().toLocaleTimeString('fa-IR')}</div>
    `;
    chatBox.appendChild(botMsg);
    
    // پاک کردن عکس بعد از ارسال
    removeImage();
    scrollToBottom();
}

// ============ اسکرول به پایین ============
function scrollToBottom() {
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
}

// ============ ارسال با Ctrl+Enter ============
messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.ctrlKey) {
        e.preventDefault();
        sendMessage();
    }
});

// ============ مقداردهی اولیه ============
document.addEventListener('DOMContentLoaded', function() {
    // تنظیم تایمستamp خوش‌آمدگویی
    document.getElementById('welcomeTime').textContent = 
        new Date().toLocaleTimeString('fa-IR');
    
    // بررسی وضعیت سرور
    checkServerStatus();
    
    // بررسی دوره‌ای هر ۳۰ ثانیه
    setInterval(checkServerStatus, 30000);
});

// ============ نمایش راهنما ============
function showHelp() {
    const helpText = `
        🩺 راهنمای استفاده:
        
        1️⃣ عکس آندوسکوپی خود را آپلود کنید
        2️⃣ منتظر بمانید تا ResNet50 تصویر را تحلیل کند
        3️⃣ نتیجه با درصد اطمینان نمایش داده می‌شود
        4️⃣ Llama 3.2 نتیجه را به زبان ساده توضیح می‌دهد
        
        ⚕️ توجه: این ابزار فقط کمکی است و تشخیص نهایی بر عهده پزشک متخصص می‌باشد.
    `;
    alert(helpText);
}

// ============ پاک کردن تاریخچه ============
function clearChat() {
    if (confirm('آیا تاریخچه گفتگو پاک شود؟')) {
        chatBox.innerHTML = `
            <div class="message bot-message">
                <div class="message-content">
                    👋 تاریخچه پاک شد. دوباره شروع کنید!
                </div>
                <div class="timestamp">${new Date().toLocaleTimeString('fa-IR')}</div>
            </div>
        `;
    }
}
