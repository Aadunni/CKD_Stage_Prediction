from fastapi import HTTPException

class ModelNotLoadedException(HTTPException):
    def __init__(self):
        super().__init__(status_code=500, detail="Model not loaded")

class PredictionException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=400, detail=f"Prediction error: {detail}")

class ValidationException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(status_code=422, detail=f"Validation error: {detail}")