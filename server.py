from fastapi import FastAPI, UploadFile, File,Request
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="templates")
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import torchvision
from torch.autograd import Variable
import PIL.ImageOps
import os
from tqdm import tqdm
import uuid

app = FastAPI()
IMAGEDIR = 'imagesnew/'
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

@app.get("/")
async def read_index():
    return FileResponse('index.html')
# Define a Pydantic model for receiving file uploads
class FileUpload(BaseModel):
    file: UploadFile

@app.post("/uploadPicture")
async def UploadImage(request: Request, mypic1: UploadFile = File(...)):
    mypic1.filename = f"{uuid.uuid4()}.png"
    contents1 = mypic1.file.read()
    with open(f"{IMAGEDIR}{mypic1.filename}", "wb") as f:
        f.write(contents1)
    
    # Process the uploaded image
    image = Image.open(io.BytesIO(contents1))
    image = image.resize((256, 256)).convert('L')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make predictions using your model
    predicted_probs = model.predict(image_array)
    predicted_label_index = np.argmax(predicted_probs)
    predicted_label = class_labels[predicted_label_index]
    # print(predicted_probs)
    # return templates.TemplateResponse("result.html", {"request": request, "predicted_label": predicted_label, "probabilities": predicted_probs,"class_labels":class_labels})

    # Process the predicted label and probabilities
    result_message = ""
    if predicted_label == 'F':
        result_message = "Case F: Handle something for label F"
        # return result_message
        # return FileResponse('s.html')
        return FileResponse('f.html')

    elif predicted_label == 'M':
        result_message = "Case M: Handle something for label M"
        # Additional logic for case M
        # return result_message
        return FileResponse('m.html')

    elif predicted_label == 'N':
        result_message = "Case n: Handle something for label N"
        # Additional logic for case M
        # return result_message
        return FileResponse('n.html')

    elif predicted_label == 'V':
        result_message = "Case v: Handle something for label V"
        # Additional logic for case M
        # return result_message
        return FileResponse('v.html')
    
    elif predicted_label == 'S':
        result_message = "Case s: Handle something for label S"
        # Additional logic for case M
        # return result_message
        return FileResponse('s.html')

    elif predicted_label == 'Q':
        result_message = "Case q: Handle something for label Q"
        # Additional logic for case M
        # return result_message
        return FileResponse('q.html')


    # Add more cases as needed

    # Choose the case with the highest probability
    highest_probability_index = np.argmax(predicted_probs)
    highest_probability_label = class_labels[highest_probability_index]
    highest_probability_value = predicted_probs[0, highest_probability_index]
    result_message += f"\nHighest Probability Case: {highest_probability_label} with Probability: {highest_probability_value:.4f}"

    # return templates.TemplateResponse("result.html", {
    #     "request": request,
    #     "predicted_label": predicted_label,
    #     "probabilities": predicted_probs,
    #     "class_labels": class_labels,
    #     "result_message": result_message
    # })

# @app.post("/uploadPicture1")
# async def UploadImage(mypic1:UploadFile=File(...)):
#     mypic1.filename = f"{uuid.uuid4()}.png"
#     contents1 = mypic1.file.read()
#     with open(f"{IMAGEDIR}{mypic1.filename}","wb") as f:
#         f.write(contents1)
#     image1_data = Image.open('imagesnew/' + mypic1.filename)
    
#     # Process the uploaded image
#     image = Image.open(io.BytesIO(image1_data))
#     image = image.resize((256, 256)).convert('L')
#     image_array = np.array(image) / 255.0
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

#     # Make predictions using your model
#     predicted_probs = model.predict(image_array)
#     predicted_label_index = np.argmax(predicted_probs)
#     predicted_label = class_labels[predicted_label_index]

#     return {"predicted_label": predicted_label, "probabilities": predicted_probs.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
  
