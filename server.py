from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from keras.models import load_model

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (adjust in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

# Load your trained model
model = load_model('your_model.h5')  # Replace with the path to your trained model

# Define the class labels
class_labels = ['F', 'M', 'N', 'Q', 'S', 'V']

# Define a Pydantic model for receiving file uploads
class FileUpload(BaseModel):
    file: UploadFile

@app.post("/predict/")
async def predict_ecg_class(file_upload: FileUpload):
    # Process the uploaded image
    image = Image.open(io.BytesIO(await file_upload.file.read()))
    image = image.resize((256, 256)).convert('L')
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
  
