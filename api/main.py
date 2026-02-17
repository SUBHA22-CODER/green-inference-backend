"""
Green-Inference: FastAPI Backend
REST API for Green-Inference orchestrator
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import yaml
from loguru import logger

from orchestrator import GreenInferenceOrchestrator, OrchestrationEvent


# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="Green-Inference API",
    description="Power-aware AI inference orchestrator",
    version="1.0.0"
)

# CORS middleware - Updated for Lovable.dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.lovable.app",
        "https://*.lovable.dev",
        "*"  # Allow all origins (for development only)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[GreenInferenceOrchestrator] = None
recent_events = []


# Request/Response models
class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None


class InferenceResponse(BaseModel):
    response: str
    model_used: str
    inference_time: float
    power_state: str
    battery_percent: float


class StatusResponse(BaseModel):
    is_running: bool
    current_model: Optional[str]
    power_state: str
    battery_percent: float
    is_plugged: bool
    total_power: float
    energy_saved_wh: float
    carbon_saved_kg: float
    inference_count: int
    model_swaps: int


# Event callback
def event_callback(event: OrchestrationEvent):
    """Store recent events for API access"""
    recent_events.append({
        'type': event.event_type,
        'timestamp': event.timestamp,
        'details': event.details
    })
    
    # Keep only last 50 events
    if len(recent_events) > 50:
        recent_events.pop(0)


@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    
    logger.info("Starting Green-Inference API server")
    
    orchestrator = GreenInferenceOrchestrator(config)
    orchestrator.register_event_callback(event_callback)
    orchestrator.start()
    
    logger.success("API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator
    
    logger.info("Shutting down Green-Inference API server")
    
    if orchestrator:
        orchestrator.stop()
    
    logger.success("API server stopped successfully")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Green-Inference API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orchestrator and orchestrator.is_running:
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=503, detail="Orchestrator not running")


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_status()
    return StatusResponse(**status)


@app.post("/inference", response_model=InferenceResponse)
async def generate_inference(request: InferenceRequest):
    """Generate AI inference"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if not orchestrator.is_running:
        raise HTTPException(status_code=503, detail="Orchestrator not running")
    
    try:
        import time
        start_time = time.time()
        
        # Generate response
        response = orchestrator.generate(request.prompt, request.max_tokens)
        
        inference_time = time.time() - start_time
        
        # Get current status
        status = orchestrator.get_status()
        
        return InferenceResponse(
            response=response,
            model_used=status['current_model'] or "Unknown",
            inference_time=inference_time,
            power_state=status['power_state'],
            battery_percent=status['battery_percent']
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List available model profiles"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    profiles = orchestrator.model_manager.list_profiles()
    return {"models": profiles}


@app.get("/events")
async def get_events(limit: int = 10):
    """Get recent orchestration events"""
    return {"events": recent_events[-limit:]}


@app.get("/metrics")
async def get_metrics():
    """Get detailed metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    status = orchestrator.get_status()
    model_stats = orchestrator.model_manager.get_stats()
    energy_stats = orchestrator.power_monitor.get_energy_stats()
    
    return {
        "power": {
            "battery_percent": status['battery_percent'],
            "is_plugged": status['is_plugged'],
            "cpu_power": status['cpu_power'],
            "gpu_power": status['gpu_power'],
            "total_power": status['total_power']
        },
        "model": {
            "current": status['current_model'],
            "load_time": model_stats['model_load_time'],
            "inference_count": model_stats['inference_count'],
            "avg_inference_time": model_stats['avg_inference_time'],
            "swap_count": model_stats['swap_count']
        },
        "energy": {
            "saved_wh": energy_stats['energy_saved_wh'],
            "runtime_hours": energy_stats['runtime_hours'],
            "carbon_saved_kg": energy_stats['carbon_saved_kg'],
            "avg_power_saved_w": energy_stats['avg_power_saved_w']
        }
    }


@app.post("/start")
async def start_orchestrator():
    """Manually start the orchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if orchestrator.is_running:
        return {"message": "Orchestrator already running"}
    
    orchestrator.start()
    return {"message": "Orchestrator started"}


@app.post("/stop")
async def stop_orchestrator():
    """Manually stop the orchestrator"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    if not orchestrator.is_running:
        return {"message": "Orchestrator already stopped"}
    
    orchestrator.stop()
    return {"message": "Orchestrator stopped"}


if __name__ == "__main__":
    import uvicorn
    
    api_config = config.get('API', {})
    host = api_config.get('HOST', '127.0.0.1')
    port = api_config.get('PORT', 8000)
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
