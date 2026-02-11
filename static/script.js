/* webapp/static/script.js */

// ============ تنظیمات اولیه ============
const chatBox = document.getElementById('chatBox');
const loading = document.getElementById('loading');
let currentImage = null;

// ============ پیش‌نمایش عکس ============
document.getElementById('imageInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // چک کردن حجم فایل (حداکثر 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('حجم فایل نباید بیشتر از 10 مگابایت باشد');
            return;
        }
        
        // چک کردن فرمت
        if (!file.type.startsWith('image/')) {
            showError('لطفاً فقط فایل تصویری آپلود کنید');
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            currentImage = e.target.result;
            document.getElementById('imagePreviewContainer').innerHTML = `
                <div class="image-preview-wrapper">
                    <img src="${e.target.result}" class="image-preview" alt="پیش‌نمایش">
                    <button onclick="removeImage()" class="remove-btn" title="حذف عکس">✕</button>
                    <span class="confidence-badge" style="position: absolute; bottom: -10px; left: 10px; background: #4299e1;">
                        🖼️ آماده تحلیل
                    </span>
                </div>
            `;
        };
        reader.readAsDataURL(file);
    }
});

// ============ حذف عکس ============
function removeImage() {
    currentImage = null;
    document.getElementById('imagePreviewContainer').innerHTML = '';
    document.getElementById('imageInput').value = '';
}

// ============ ارسال پیام ============
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message && !currentImage) {
        showError('لطفاً پیام بنویسید یا عکس آپلود کنید');
        return;
    }
    
    // نمایش پیام کاربر
    addMessage(message || '🖼️ تحلیل تصویر', 'user');
    
    // نمایش عکس در چت
    if (currentImage) {
        addImageMessage(currentImage, 'user');
    }
    
    // پاک کردن ورودی
    messageInput.value = '';
    
    // نمایش لودینگ
    showLoading();
    
    try {
        // ارسال به سرور
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                image: currentImage
            })
        });
        
        const data = await response.json();
        hideLoading();
        
        if (data.error) {
            showError(data.error);
        } else {
            // نمایش نتیجه پیش‌بینی
            showPredictionResult(data);
        }
        
        // پاک کردن عکس بعد از ارسال (اختیاری)
        // removeImage();
        
    } catch (error) {
        hideLoading();
        showError('خطا در ارتباط با سرور: ' + error.message);
    }
}

// ============ نمایش نتیجه پیش‌بینی ============
function showPredictionResult(data) {
    let confidenceColor = '';
    let confidenceText = '';
    
    if (data.confidence > 0.9) {
        confidenceColor = '#48bb78';
        confidenceText = 'بسیار مطمئن';
    } else if (data.confidence > 0.7) {
        confidenceColor = '#ecc94b';
        confidenceText = 'مطمئن';
    } else {
        confidenceColor = '#f56565';
        confidenceText = 'کمتر مطمئن';
    }
    
    const resultHTML = `
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <span style="font-size: 24px;">🔬</span>
            <span style="font-weight: 700; color: #2d3748;">نتیجه تشخیص:</span>
        </div>
        
        <div style="background: linear-gradient(135deg, #667eea15, #764ba215); padding: 20px; border-radius: 15px; border-right: 5px solid #667eea;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                <div>
                    <span style="font-size: 32px; font-weight: 800; color: #4a5568;">
                        ${data.class_fa}
                    </span>
                    <span style="display: inline-block; margin-right: 10px; padding: 5px 15px; background: ${confidenceColor}; color: white; border-radius: 20px; font-size: 12px; font-weight: 600;">
                        ${data.confidence_percent} - ${confidenceText}
                    </span>
                </div>
            </div>
            
            <div style="background: white; padding: 15px; border-radius: 10px; margin-top: 10px;">
                <div style="display: flex; gap: 15px; align-items: center;">
                    <div style="width: 60px; height: 60px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 30px;">📊</span>
                    </div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #718096; font-size: 13px;">درصد اطمینان</span>
                            <span style="font-weight: 700; color: #2d3748;">${data.confidence_percent}</span>
                        </div>
                        <div style="width: 100%; height: 8px; background: #edf2f7; border-radius: 4px;">
                            <div style="width: ${data.confidence_percent}; height: 8px; background: linear-gradient(90deg, #48bb78, #4299e1); border-radius: 4px; transition: width 0.5s;"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background: #fff3e0; border-radius: 10px; border-right: 4px solid #ed8936;">
                <div style="display: flex; gap: 10px;">
                    <span style="font-size: 20px;">💡</span>
                    <div>
                        <span style="font-weight: 700; color: #2d3748; display: block; margin-bottom: 5px;">
                            توضیح تشخیص:
                        </span>
                        <span style="color: #4a5568; line-height: 1.6;">
                            ${data.explanation}
                        </span>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 15px; display: flex; gap: 10px; justify-content: flex-end;">
                <span style="padding: 5px 12px; background: #e2e8f0; border-radius: 15px; font-size: 12px; color: #4a5568;">
                    ⚕️ تشخیص هوش مصنوعی
                </span>
                <span style="padding: 5px 12px; background: #e2e8f0; border-radius: 15px; font-size: 12px; color: #4a5568;">
                    🏥 نیاز به تأیید پزشک
                </span>
            </div>
        </div>
    `;
    
    addMessage(resultHTML, 'bot');
}

// ============ اضافه کردن پیام متنی ============
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.style.animation = 'fadeIn 0.3s ease';
    
    const timestamp = new Date().toLocaleTimeString('fa-IR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${content}
            ${sender === 'bot' ? '<div style="margin-top: 10px; font-size: 12px; color: #a0aec0;">🤖 دستیار هوشمند</div>' : ''}
        </div>
        <div class="timestamp">${timestamp}</div>
    `;
    
    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

// ============ اضافه کردن پیام تصویری ============
function addImageMessage(imageData, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.style.animation = 'fadeIn 0.3s ease';
    
    const timestamp = new Date().toLocaleTimeString('fa-IR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-content" style="padding: 10px; max-width: 300px;">
            <img src="${imageData}" style="width: 100%; border-radius: 10px; border: 2px solid #4299e1;">
            <div style="margin-top: 5px; font-size: 11px; color: #718096; text-align: center;">
                🖼️ تصویر آپلود شده
            </div>
        </div>
        <div class="timestamp">${timestamp}</div>
    `;
    
    chatBox.appendChild(messageDiv);
    scrollToBottom();
}

// ============ نمایش خطا ============
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
        background: #fed7d7;
        color: #c53030;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-right: 4px solid #f56565;
        animation: fadeIn 0.3s ease;
    `;
    errorDiv.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 20px;">❌</span>
            <span style="font-weight: 500;">${message}</span>
        </div>
    `;
    
    chatBox.appendChild(errorDiv);
    scrollToBottom();
    
    // پاک کردن خودکار بعد از 5 ثانیه
    setTimeout(() => {
        errorDiv.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

// ============ نمایش لودینگ ============
function showLoading() {
    loading.style.display = 'block';
    chatBox.style.opacity = '0.7';
}

function hideLoading() {
    loading.style.display = 'none';
    chatBox.style.opacity = '1';
}

// ============ اسکرول به پایین ============
function scrollToBottom() {
    chatBox.scrollTo({
        top: chatBox.scrollHeight,
        behavior: 'smooth'
    });
}

// ============ دکمه اسکرول به بالا/پایین ============
function createScrollButton() {
    const scrollBtn = document.createElement('button');
    scrollBtn.className = 'scroll-btn';
    scrollBtn.innerHTML = '⬇️';
    scrollBtn.onclick = () => scrollToBottom();
    document.body.appendChild(scrollBtn);
    
    window.addEventListener('scroll', () => {
        if (window.scrollY > 200) {
            scrollBtn.style.display = 'flex';
        } else {
            scrollBtn.style.display = 'none';
        }
    });
}

// ============ نمایش راهنما ============
function showHelp() {
    const helpHTML = `
        <div style="background: #ebf8ff; padding: 20px; border-radius: 15px; margin-bottom: 20px; border-right: 4px solid #4299e1;">
            <div style="display: flex; gap: 15px; align-items: center; margin-bottom: 15px;">
                <span style="font-size: 28px;">🆘</span>
                <span style="font-weight: 700; color: #2c5282; font-size: 18px;">راهنمای استفاده</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <span style="font-size: 24px; display: block; margin-bottom: 5px;">📤</span>
                    <span style="font-weight: 600; color: #2d3748;">۱. آپلود عکس</span>
                    <p style="color: #718096; font-size: 13px; margin-top: 5px;">عکس آندوسکوپی رو آپلود کن</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <span style="font-size: 24px; display: block; margin-bottom: 5px;">💬</span>
                    <span style="font-weight: 600; color: #2d3748;">۲. سوال بپرس</span>
                    <p style="color: #718096; font-size: 13px; margin-top: 5px;">مثلاً: این عکس رو تحلیل کن</p>
                </div>
                <div style="background: white; padding: 15px; border-radius: 10px;">
                    <span style="font-size: 24px; display: block; margin-bottom: 5px;">🔬</span>
                    <span style="font-weight: 600; color: #2d3748;">۳. دریافت نتیجه</span>
                    <p style="color: #718096; font-size: 13px; margin-top: 5px;">تشخیص + درصد اطمینان</p>
                </div>
            </div>
            <p style="color: #4a5568; margin-top: 15px; font-size: 13px; background: #fff3cd; padding: 10px; border-radius: 8px;">
                ⚠️ توجه: این سیستم فقط یک ابزار کمکی است و تشخیص نهایی بر عهده پزشک متخصص می‌باشد.
            </p>
        </div>
    `;
    
    addMessage(helpHTML, 'bot');
}

// ============ پاک کردن تاریخچه چت ============
function clearChat() {
    if (confirm('آیا می‌خواهید تاریخچه چت را پاک کنید؟')) {
        chatBox.innerHTML = `
            <div class="message bot-message">
                <div class="message-content">
                    👋 سلام! من دستیار تشخیص کرون و کولیت اولسراتیو هستم.
                    <br><br>
                    📤 می‌تونی عکس آندوسکوپی رو آپلود کنی و سوالاتت رو بپرسی.
                    <br>
                    🧠 من با مدل ResNet50 آموزش دیدم و دقت بالای ۹۰٪ دارم.
                </div>
                <div class="timestamp">${new Date().toLocaleTimeString('fa-IR')}</div>
            </div>
        `;
        removeImage();
    }
}

// ============ ذخیره تاریخچه چت ============
function saveChat() {
    const chatHistory = chatBox.innerHTML;
    const blob = new Blob([chatHistory], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat_history_${new Date().toISOString().slice(0,10)}.html`;
    a.click();
}

// ============ رویدادهای صفحه ============
document.addEventListener('DOMContentLoaded', function() {
    // نمایش راهنما
    setTimeout(() => showHelp(), 500);
    
    // ایجاد دکمه اسکرول
    createScrollButton();
    
    // ارسال با Ctrl+Enter
    document.getElementById('messageInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Drag & Drop برای آپلود عکس
    const dropZone = document.querySelector('.input-area');
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.background = '#ebf8ff';
        dropZone.style.border = '2px dashed #4299e1';
    });
    
    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.style.background = '';
        dropZone.style.border = 'none';
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.background = '';
        dropZone.style.border = 'none';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            document.getElementById('imageInput').files = e.dataTransfer.files;
            // trigger change event
            const event = new Event('change', { bubbles: true });
            document.getElementById('imageInput').dispatchEvent(event);
        }
    });
});

// ============ انیمیشن‌ها ============
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }
    
    .message-content a {
        color: #4299e1;
        text-decoration: none;
    }
    
    .message-content a:hover {
        text-decoration: underline;
    }
    
    .typing-indicator {
        display: flex;
        gap: 5px;
        padding: 10px 15px;
        background: white;
        border-radius: 20px;
        border-bottom-left-radius: 5px;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #a0aec0;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); background: #4299e1; }
    }
`;

document.head.appendChild(style);