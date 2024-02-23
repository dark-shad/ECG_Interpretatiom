from fastapi import FastAPI, UploadFile, File, Request
from starlette.responses import FileResponse, HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from scipy.signal import find_peaks
import cv2
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Load your trained model
model = load_model('ECG_MODEL.h5')  # Replace with the path to your trained model

# Define the class labels
class_labels = ['F', 'M', 'N', 'Q', 'S', 'V']

# Function for R peak detection
def detect_peaks(signal, threshold):
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/uploadPicture")
async def upload_and_classify_beats(request: Request, mypic1: UploadFile = File(...)):
    contents1 = await mypic1.read()
    image = Image.open(io.BytesIO(contents1)).convert('L')  # Convert to grayscale
    image_array = np.array(image)

    # Preprocessing
    image_blurred = cv2.GaussianBlur(image_array, (7, 7), 0)

    # Edge detection and segmentation
    edges = cv2.Canny(image_blurred, threshold1=50, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour corresponds to the ECG waveform
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the ECG waveform
    mask = np.zeros_like(image_array)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # R peak detection
    vertical_projection = np.sum(mask, axis=0)
    r_peak_indices = detect_peaks(vertical_projection, threshold=150)

    # Beat classification
    beat_width = 120
    beat_labels = []
    beats = []
    for r_peak_index in r_peak_indices:
        # beat_roi = mask[:, max(0, r_peak_index - beat_width):min(r_peak_index + beat_width, mask.shape[1])]
        start_index = max(0, r_peak_index - beat_width // 2)
        end_index = min(r_peak_index + beat_width // 2, len(mask[0]))
        beat_roi = mask[:, start_index:end_index]
        beats.append(beat_roi)
        # Resize beat to match model input size
        beat_roi_resized = cv2.resize(beat_roi, (256, 256))
        beat_roi_normalized = beat_roi_resized / 255.0
        beat_roi_expanded = np.expand_dims(beat_roi_normalized, axis=0)  # Add batch dimension

        # Make predictions using your model
        predicted_probs = model.predict(beat_roi_expanded)
        predicted_label_index = np.argmax(predicted_probs)
        predicted_label = class_labels[predicted_label_index]
        beat_labels.append(predicted_label)
    # Display the original image and extracted beats for visualization
    
    
    # Construct the HTML response manually
    html_content = "<h1>ECG Beat Classification Result</h1>"
    html_content += "<h2>Predicted Beat Labels:</h2>"
    html_content += "<ul>"
    for label in beat_labels:
        html_content += f"<li>{label}</li>"
    html_content += "</ul>"
    
    return HTMLResponse(content=html_content, status_code=200)
    
    # cv2.imshow('Original ECG Image', ecg_image)
    # for i, beat in enumerate(beats):
    #     cv2.imshow(f'Extracted Beat {i+1}', beat)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
