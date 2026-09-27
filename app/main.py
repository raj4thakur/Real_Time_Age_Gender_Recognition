from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from pathlib import Path
from fastapi.responses import FileResponse

from app.inference import predict_age_gender



app = FastAPI(title="Age and Gender Prediction API", description="Real-time age and gender recognition using computer vision.", version="1.0.0")

class prediction_response(BaseModel):
    age: int
    gender: str
    confidence: float


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
@app.get("/")
async def home():
    return FileResponse(FRONTEND_FILE)


@app.get("/Health")
async def health_check():
    return {"status":"API is healthy and running."}


@app.post("/predict", response_model=prediction_response)
async def predict(file: UploadFile = File(...)):
    #check content type
    
    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Could not determine file type."
        )
    if file.content_type not in ["image/jpeg", "image/png"]:
        return {"error": "Invalid file type. Please upload a JPEG or PNG image."}
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported."
        )
    image_bytes =await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )
    
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please upload a valid JPEG or PNG image."
        )
    
    try:
        result = predict_age_gender(image)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model Inference Error: {str(e)}"
        )
    return result
    
