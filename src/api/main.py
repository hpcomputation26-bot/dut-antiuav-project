from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil, uuid, os, tempfile
from model_wrapper import DetectorWrapper

app = FastAPI(title="DUT Anti-UAV Detection API")
detector = DetectorWrapper(weights_path="best.pt")

@app.get("/")
def root():
    return {"message": "DUT Anti-UAV Drone Detection API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        detections = detector.predict(temp_path)
        return JSONResponse({"detections": detections})
    finally:
        os.remove(temp_path)
