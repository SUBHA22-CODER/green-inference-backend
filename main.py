from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Green Inference Backend")

class InferenceRequest(BaseModel):
    model_name: str
    input_tokens: int

@app.get("/")
def root():
    return {"status": "Green Inference Backend Running 🚀"}

@app.post("/predict")
def predict(request: InferenceRequest):
    # Simulated carbon estimation
    estimated_power = request.input_tokens * 0.002
    carbon_emission = estimated_power * 0.4
    
    return {
        "model": request.model_name,
        "estimated_power_watts": round(estimated_power, 4),
        "carbon_emission_kg": round(carbon_emission, 6),
        "mode": "Simulation Mode"
    }
