
from pathlib import Path
import base64
import logging

import cv2
import numpy as np
import uvicorn

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO


# ==========================================
# 1. تنظیم مسیرهای پروژه و لاگینگ
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
TEMPLATE_DIR = BASE_DIR / "templates"

# حداکثر حجم فایل آپلود (20 مگابایت)
MAX_FILE_SIZE = 20 * 1024 * 1024

app = FastAPI(title="وب‌اپ تشخیص آتش")

templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# ==========================================
# 2. بارگذاری مدل YOLO
# ==========================================

model = None
model_class_names = {}

if MODEL_PATH.is_file():
    try:
        model = YOLO(str(MODEL_PATH))
        logger.info("✓ مدل YOLO با موفقیت بارگذاری شد.")
        
        # استخراج نام کلاس‌ها برای استفاده بعدی
        if hasattr(model, 'names'):
            model_class_names = model.names
            logger.info(f"کلاس‌های مدل: {model_class_names}")
        
    except Exception as e:
        logger.error(f"✗ خطا در بارگذاری مدل: {e}")
        model = None

else:
    logger.warning(f"✗ فایل مدل پیدا نشد: {MODEL_PATH}")


# ==========================================
# 3. صفحه اصلی وب‌اپ
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ==========================================
# 4. توابع کمکی اعتبارسنجی
# ==========================================

def validate_image_data(raw_data: bytes, context: str = "image") -> np.ndarray:
    """
    اعتبارسنجی و تبدیل داده‌های خام تصویر به آرایه NumPy
    """
    if not raw_data:
        logger.warning(f"داده خالی دریافت شد: {context}")
        raise HTTPException(
            status_code=400,
            detail="فایل انتخاب‌شده خالی است."
        )
    
    if len(raw_data) > MAX_FILE_SIZE:
        logger.warning(f"فایل بیش از حد بزرگ: {len(raw_data)} bytes")
        raise HTTPException(
            status_code=400,
            detail=f"حجم فایل نباید بیشتر از {MAX_FILE_SIZE // (1024*1024)} مگابایت باشد."
        )
    
    try:
        image_array = np.frombuffer(raw_data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"تبدیل تصویر ناموفق بود: {context}")
            raise HTTPException(
                status_code=400,
                detail="فایل انتخاب‌شده یک تصویر معتبر نیست."
            )
        
        # بررسی ابعاد معقول تصویر
        h, w = image.shape[:2]
        if h < 10 or w < 10 or h > 10000 or w > 10000:
            logger.warning(f"ابعاد غیرمعمول تصویر: {w}x{h}")
            raise HTTPException(
                status_code=400,
                detail="ابعاد تصویر نامعتبر است."
            )
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطا در پردازش تصویر ({context}): {e}")
        raise HTTPException(
            status_code=400,
            detail="خواندن تصویر امکان‌پذیر نیست."
        )


# ==========================================
# 5. تابع تشخیص آتش
# ==========================================

def run_detection(image: np.ndarray):
    """
    اجرای مدل تشخیص روی تصویر ورودی
    """
    if model is None:
        logger.error("تلاش برای استفاده از مدل بارگذاری‌نشده")
        raise HTTPException(
            status_code=503,
            detail="مدل بارگذاری نشده است. فایل best.pt را بررسی کنید."
        )

    try:
        results = model.predict(
            source=image,
            conf=0.50,
            verbose=False
        )

        if not results or len(results) == 0:
            logger.warning("مدل نتیجه‌ای برنگرداند")
            raise HTTPException(
                status_code=500,
                detail="مدل نتوانست تصویر را پردازش کند."
            )

        result = results[0]
        
        # بررسی وجود متد plot
        if not hasattr(result, 'plot'):
            logger.error("نسخه مدل plot را پشتیبانی نمی‌کند")
            raise HTTPException(
                status_code=500,
                detail="نسخه مدل با سرور سازگار نیست."
            )
            
        annotated_image = result.plot()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطا در اجرای مدل: {e}")
        raise HTTPException(
            status_code=500,
            detail="خطا در اجرای مدل. لطفاً دوباره تلاش کنید."
        )

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        for box in result.boxes:

            try:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                coordinates = [
                    int(value)
                    for value in box.xyxy[0].tolist()
                ]

                class_name = result.names.get(
                    class_id,
                    str(class_id)
                )

                detections.append({
                    "class_id": class_id,
                    "name": str(class_name),
                    "confidence": round(confidence, 4),
                    "confidence_percent": round(
                        confidence * 100, 1
                    ),
                    "box": coordinates
                })
                
            except Exception as e:
                logger.warning(f"خطا در پردازش یک detection: {e}")
                continue

    # تبدیل تصویر حاشیه‌نویسی‌شده به base64
    try:
        success, buffer = cv2.imencode(".jpg", annotated_image)

        if not success:
            logger.error("تبدیل تصویر به JPG ناموفق بود")
            raise HTTPException(
                status_code=500,
                detail="تبدیل تصویر نتیجه به JPG ناموفق بود."
            )

        encoded_image = base64.b64encode(
            buffer.tobytes()
        ).decode("ascii")
        
    except Exception as e:
        logger.error(f"خطا در رمزگذاری تصویر: {e}")
        raise HTTPException(
            status_code=500,
            detail="خطا در تولید تصویر نتیجه."
        )

    # بررسی وجود کلاس آتش در نتایج
    # جستجوی robust برای کلاس fire
    fire_detected = any(
        "fire" in item["name"].lower()
        or "آتش" in item["name"].lower()
        for item in detections
    )
    
    if detections:
        logger.info(f"تشخیص انجام شد: {len(detections)} شی، آتش: {fire_detected}")

    return {
        "image": f"data:image/jpeg;base64,{encoded_image}",
        "detections": detections,
        "count": len(detections),
        "fire_detected": fire_detected
    }


# ==========================================
# 6. دریافت عکس و تشخیص آتش
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    دریافت تصویر آپلودشده و اجرای تشخیص
    """
    # اعتبارسنجی نوع محتوا
    if file.content_type and not file.content_type.startswith("image/"):
        logger.warning(f"نوع فایل نامعتبر: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="لطفاً یک فایل تصویری انتخاب کنید."
        )
    
    # اعتبارسنجی نام فایل
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="نام فایل مشخص نیست."
        )

    raw_data = await file.read()
    
    logger.info(f"دریافت فایل: {file.filename}, حجم: {len(raw_data)} bytes")
    
    image = validate_image_data(raw_data, context=f"upload:{file.filename}")

    return run_detection(image)


# ==========================================
# 7. دریافت فریم دوربین مرورگر
# ==========================================

@app.post("/predict-frame")
async def predict_frame(file: UploadFile = File(...)):
    """
    دریافت فریم از دوربین مرورگر و اجرای تشخیص
    """
    raw_data = await file.read()
    
    image = validate_image_data(raw_data, context="camera-frame")

    return run_detection(image)


# ==========================================
# 8. بررسی وضعیت برنامه و مدل
# ==========================================

@app.get("/health")
async def health():
    """
    بررسی وضعیت سلامت برنامه و مدل
    """
    is_ready = model is not None
    
    response = {
        "status": "ready" if is_ready else "unavailable",
        "model_loaded": is_ready,
    }
    
    # افزودن اطلاعات کلاس‌های مدل اگر مدل بارگذاری شده باشد
    if is_ready and model_class_names:
        response["model_classes"] = list(model_class_names.values())
    
    logger.info(f"Health check: {response['status']}")
    
    return response


# ==========================================
# 9. اجرای مستقیم برنامه با python main.py
# ==========================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        reload=False
    )
