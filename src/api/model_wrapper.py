from ultralytics import RTDETR

class DetectorWrapper:
    def __init__(self, weights_path="best.pt"):
        self.model = RTDETR(weights_path)

    def predict(self, image_path, conf=0.3):
        results = self.model.predict(image_path, conf=conf)
        boxes = results[0].boxes
        return [
            {
                "bbox": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class": "UAV"
            }
            for box in boxes
        ]
