# Fire Detection Web Application 🔥

A modern, bilingual (Persian/English) web application for real-time fire detection using YOLOv8 deep learning model. Built with FastAPI backend and vanilla JavaScript frontend.

## Features ✨

### Core Functionality
- **Image Analysis**: Upload images for fire detection analysis
- **Live Camera Detection**: Real-time detection using browser camera
- **YOLO Model Integration**: Powered by Ultralytics YOLOv8
- **Detection Results**: Annotated images with bounding boxes and confidence scores

### User Experience
- **Bilingual Interface**: Full Persian (RTL) and English (LTR) support
- **Dark/Light Themes**: User-selectable themes with system preference detection
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Accessible**: Keyboard navigation, ARIA labels, and semantic HTML
- **Professional UI**: Clean, modern design with smooth transitions

### Technical Features
- **Robust Input Validation**: File type, size, and image dimension checks
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Secure**: No sensitive path exposure, proper logging, input sanitization
- **Camera Management**: Safe camera start/stop with permission handling
- **Non-blocking Processing**: Sequential frame processing prevents overload

## Architecture 🏗️

```
fire_webapp/
├── main.py              # FastAPI backend server
├── best.pt              # YOLOv8 trained model (you need to provide this)
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html      # Single-page frontend application
└── README.md           # This file
```

## Requirements 📋

- Python 3.8+
- Modern web browser with camera support
- Trained YOLOv8 model file (`best.pt`)

## Installation 🚀

### 1. Clone or Download the Project

```bash
cd fire_webapp
```

### 2. Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Model

Place your trained `best.pt` YOLOv8 model file in the `fire_webapp/` directory (same location as `main.py`).

## Usage 🎯

### Start the Server

```bash
python main.py
```

The server will start at: `http://127.0.0.1:8002`

### Access the Application

Open your browser and navigate to:
```
http://127.0.0.1:8002
```

### Image Analysis Mode

1. Click the "Image Analysis" tab
2. Click "Select Image" and choose an image file
3. Preview will appear automatically
4. Click "Start Detection" to analyze
5. View results with annotated image and detection details
6. Click "Clear" to reset

### Live Camera Mode

1. Click the "Live Camera" tab
2. Click "Start Camera" and approve browser permission
3. Camera will start and process frames automatically
4. Click "Stop Camera" to end detection
5. Switch modes as needed

### Interface Controls

- **Language Toggle**: Switch between Persian and English (top-right)
- **Theme Toggle**: Switch between light and dark mode (top-right)
- **Mode Tabs**: Switch between image and camera modes

## API Endpoints 🔌

### `GET /`
Returns the main application interface (HTML).

### `POST /predict`
Analyze uploaded image for fire detection.

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (image file)

**Response:**
```json
{
  "image": "data:image/jpeg;base64,...",
  "detections": [
    {
      "class_id": 0,
      "name": "fire",
      "confidence": 0.8524,
      "confidence_percent": 85.2,
      "box": [100, 150, 300, 400]
    }
  ],
  "count": 1,
  "fire_detected": true
}
```

### `POST /predict-frame`
Process camera frame for detection (same interface as `/predict`).

### `GET /health`
Check server and model status.

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "model_classes": ["fire", "smoke"]
}
```

## Configuration ⚙️

### Maximum Upload Size
Edit `main.py` to change the maximum file size (default: 20 MB):

```python
MAX_FILE_SIZE = 20 * 1024 * 1024  # bytes
```

### Confidence Threshold
Adjust detection confidence in `run_detection()` function:

```python
results = model.predict(
    source=image,
    conf=0.50,  # Change this value (0.0 - 1.0)
    verbose=False
)
```

### Server Port
Change the port in the `__main__` block:

```python
uvicorn.run(
    app,
    host="127.0.0.1",
    port=8002,  # Change this
    reload=False
)
```

### Camera Frame Rate
Adjust processing interval in `index.html` JavaScript:

```javascript
state.liveTimer = setInterval(processCameraFrame, 1500);  // milliseconds
```

## Security Considerations 🔒

### Implemented Security Measures

1. **Input Validation**
   - File type validation (server-side)
   - File size limits (20 MB default)
   - Image dimension validation
   - Empty file rejection

2. **Error Handling**
   - No stack trace exposure to clients
   - Sanitized error messages
   - Secure logging on server

3. **Resource Protection**
   - Maximum file size enforcement
   - Sequential frame processing
   - Automatic camera cleanup

### Recommendations for Production

⚠️ **This application is designed for local development and educational purposes.** For production deployment:

1. **Add Authentication**: Implement user authentication and authorization
2. **Use HTTPS**: Deploy behind HTTPS proxy (nginx, Apache)
3. **Rate Limiting**: Add request rate limiting to prevent abuse
4. **CORS Configuration**: Restrict allowed origins if needed
5. **Content Security Policy**: Add CSP headers
6. **File Upload Scanning**: Add virus/malware scanning for uploads
7. **Database**: Consider adding database for logging and analytics
8. **Load Balancing**: Use multiple workers for high traffic
9. **Model Versioning**: Implement model version management
10. **Monitoring**: Add application and model performance monitoring

## Testing 🧪

### Manual Testing Checklist

#### Backend Tests
- [x] Application starts with valid model
- [x] Upload and detect valid image
- [x] Reject empty files
- [x] Handle invalid image data
- [x] Return consistent detection fields
- [x] Handle images with no detections

#### Frontend Tests
- [x] Persian RTL layout displays correctly
- [x] English LTR layout displays correctly
- [x] Language preference persists after reload
- [x] Dark theme displays correctly
- [x] Light theme displays correctly
- [x] Theme preference persists after reload
- [x] System theme preference detection
- [x] Image preview before detection
- [x] Clear/reset functionality
- [x] Responsive layout on mobile
- [x] Accessible controls and navigation

### Automated Testing

You can test the health endpoint:

```bash
curl http://127.0.0.1:8002/health
```

Test image upload:

```bash
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:8002/predict
```

## Troubleshooting 🔧

### Model Not Loading

**Issue**: "Model not loaded" error or health check shows `model_loaded: false`

**Solution**:
1. Verify `best.pt` exists in `fire_webapp/` directory
2. Check file permissions
3. Ensure model is a valid YOLOv8 model
4. Check server logs for detailed error messages

### Camera Not Working

**Issue**: Camera doesn't start or permission denied

**Solution**:
1. Ensure you're using `https://` or `localhost`
2. Check browser camera permissions in settings
3. Try a different browser (Chrome recommended)
4. Verify no other application is using the camera
5. Check browser console for detailed errors

### Upload Fails

**Issue**: Image upload returns error

**Solution**:
1. Verify file is a valid image (JPG, PNG, WEBP)
2. Check file size (must be < 20 MB)
3. Try a different image
4. Check server logs for details

### Slow Detection

**Issue**: Detection takes too long

**Solution**:
1. Use smaller images (resize before upload)
2. Ensure adequate CPU/GPU resources
3. Consider using YOLOv8n (nano) model for speed
4. Check for other resource-intensive processes

### Language/Theme Not Persisting

**Issue**: Settings reset after page reload

**Solution**:
1. Check browser localStorage is enabled
2. Ensure cookies/data not being cleared
3. Try a different browser
4. Check browser privacy settings

## Known Limitations ⚠️

1. **Model Class Names**: Fire detection depends on model having "fire" or "آتش" in class names. If your model uses different naming, you'll need to modify the detection logic.

2. **No Detection ≠ Safe**: The absence of detections does NOT mean the scene is confirmed safe. This is a detection tool, not a certification system.

3. **Camera Performance**: Live camera detection may have latency based on hardware and network conditions. Not guaranteed real-time.

4. **Single Model Support**: Application loads one model at startup. Model hot-swapping not supported.

5. **No Concurrent Processing**: Camera frames are processed sequentially to prevent overload.

6. **Browser Compatibility**: Camera features require modern browsers with getUserMedia support.

## Improvements from Original Code 🎨

### Backend Improvements
- ✅ Proper logging system with structured messages
- ✅ Input validation with size and dimension checks
- ✅ Better error handling without stack trace exposure
- ✅ Model class names extraction and exposure
- ✅ Health endpoint improvements
- ✅ Filename validation
- ✅ Consistent HTTP status codes
- ✅ Exception handling in detection loop

### Frontend Improvements
- ✅ Complete UI redesign with modern aesthetics
- ✅ Bilingual support (Persian/English) with proper RTL/LTR
- ✅ Dark/light theme with persistence
- ✅ Improved camera controls and error handling
- ✅ Better status messages and visual feedback
- ✅ Responsive design for all screen sizes
- ✅ Accessible interface with ARIA labels
- ✅ Clear reset functionality
- ✅ Prevention of overlapping requests
- ✅ Better preview management
- ✅ Structured translation system
- ✅ HTML escaping for security

### User Experience
- ✅ Professional, clean design
- ✅ Smooth transitions and animations
- ✅ Clear visual hierarchy
- ✅ Better loading states
- ✅ Comprehensive error messages
- ✅ Intuitive controls

## Contributing 🤝

This is an educational project. Suggestions for improvements:

1. Add unit tests and integration tests
2. Implement model configuration UI
3. Add detection history/logging
4. Support multiple camera selection
5. Add image pre-processing options
6. Implement batch image processing
7. Add detection confidence threshold UI control
8. Export detection results (JSON, CSV)
9. Add video file upload and processing
10. Implement websocket for live updates

## License 📄

This project is for educational purposes. Ensure you have proper licensing for:
- YOLOv8 model (Ultralytics)
- Your trained model weights
- Any datasets used for training

## Disclaimer ⚠️

**This application is an assistive tool and is NOT a replacement for certified safety equipment and fire alarm systems.**

- Do not rely solely on this system for fire safety
- Always follow proper fire safety protocols
- Use certified fire detection equipment for critical applications
- This is a demonstration/educational project
- No warranty or guarantee of accuracy provided

## Support 💬

For issues and questions:
1. Check the troubleshooting section
2. Review server logs for detailed errors
3. Verify model and dependencies are correctly installed
4. Test with the `/health` endpoint

## Acknowledgments 🙏

- **Ultralytics YOLOv8**: Object detection model
- **FastAPI**: Modern Python web framework
- **OpenCV**: Computer vision library

---

**Version**: 2.0  
**Last Updated**: 2026  
