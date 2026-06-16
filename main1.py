from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
from PIL import Image
import io
import logging

from schemas import PredictionResponse, HealthResponse
from model_loader import CatDogClassifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Cat-Dog Classification API",
    description="API for classifying images as cat or dog using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable for model
model = None

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model = CatDogClassifier(model_path=r"D:/Payton/Python Packages/Software Enginnering/FasttAPI/uploads/new_model23.keras")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Cat-Dog Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for classification",
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an uploaded image contains a cat or dog
    
    Args:
        file: Image file (supports jpg, png, jpeg)
    
    Returns:
        PredictionResponse with classification result
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
    # Validate file size (5MB limit)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 5MB"
        )
    
    try:
        # Open and validate image
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        prediction_result = model.predict(image)
        
        logger.info(f"Prediction made: {prediction_result['prediction']} "
                   f"with confidence {prediction_result['confidence']:.3f}")
        
        return PredictionResponse(**prediction_result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Optional: Add batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classify multiple images in batch"""
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch request"
        )
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            prediction = model.predict(image)
            results.append({
                "filename": file.filename,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )