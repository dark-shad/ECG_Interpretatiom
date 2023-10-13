from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
import os
import shutil
from PIL import Image
import io
import numpy as np
from keras.models import load_model

app = FastAPI()

# Create a directory for uploaded ECG images
UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load your trained model
model = load_model('ECG_Model.h5')  # Replace with the path to your trained model

# Define the class labels
class_labels = ['F', 'M', 'N', 'Q', 'S', 'V']

# Serve the HTML form
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Define a Pydantic model for receiving file uploads
class FileUpload(BaseModel):
    file: UploadFile

# Handle file uploads and make predictions
@app.post("/uploadPicture")
async def upload_ecg_image(mypic1: UploadFile = File(...)):
    # Create a unique filename for the uploaded image
    unique_filename = f"{UPLOAD_DIR}/{mypic1.filename}"

    # Save the uploaded image
    with open(unique_filename, "wb") as image_file:
        shutil.copyfileobj(mypic1.file, image_file)

    # Load and process the uploaded image
    image = Image.open(unique_filename).convert('L').resize((256, 256))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions using your model
    predicted_probs = model.predict(image_array)
    predicted_label_index = np.argmax(predicted_probs)
    predicted_label = class_labels[predicted_label_index]

    return {"predicted_label": predicted_label, "probabilities": predicted_probs.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
