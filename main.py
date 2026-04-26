from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional
from env.environment import StaleMindEnv as DriftGym
import os

app = FastAPI()

# Serve assets (video, images) at /media — separate from Gradio's /assets/
_assets_dir = os.path.join(os.path.dirname(__file__), "assets")
if os.path.isdir(_assets_dir):
    app.mount("/media", StaticFiles(directory=_assets_dir), name="media")

@app.get("/video")
def serve_video():
    """Direct video endpoint — always works regardless of static file routing."""
    path = os.path.join(os.path.dirname(__file__), "assets", "siri_type.mp4")
    return FileResponse(path, media_type="video/mp4")

# Session storage
envs: Dict[str, DriftGym] = {}

def get_env(session_id: str) -> DriftGym:
    if session_id not in envs:
        envs[session_id] = DriftGym()
        envs[session_id].reset()
    return envs[session_id]



class ResetRequest(BaseModel):
    scenario_index: Optional[int] = None
    session_id: str = "default"
    config: Optional[dict] = None

class StepRequest(BaseModel):
    type: str
    content: str = ""
    session_id: str = "default"

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    env = get_env(req.session_id)
    obs, _ = env.reset(req.scenario_index, req.config)
    return {"observation": obs}

@app.post("/step")
def step(action: StepRequest):
    env = get_env(action.session_id)
    obs, reward, done, info = env.step(action.model_dump() if hasattr(action, 'model_dump') else action.dict())
    
    if obs is None:
        return {
            "observation": {},
            "reward": {"score": 0.0},
            "done": True,
            "message": "Episode complete"
        }
        
    return {
        "observation": obs,
        "reward": {"score": reward},
        "done": done,
        "info": info,
    }

@app.get("/state")
def state(session_id: str = "default"):
    return get_env(session_id).state()
