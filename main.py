from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Optional
from env.environment import StaleMindEnv as DriftGym
import os

app = FastAPI()

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
    
    # Standardize observation
    message = obs.get("message") or obs.get("request") if isinstance(obs, dict) else str(obs)
    
    return {
        "observation": {"message": message, "raw": obs},
        "reward": {"alignment": 0.0},
        "done": False
    }

@app.post("/step")
def step(action: StepRequest):
    env = get_env(action.session_id)
    obs, reward, done, info = env.step(action.model_dump() if hasattr(action, 'model_dump') else action.dict())
    
    if obs is None:
        return {
            "observation": {"message": "Episode complete"},
            "reward": {"alignment": 0.0},
            "done": True,
            "info": info
        }
        
    # Standardize observation
    message = obs.get("message") or obs.get("request") if isinstance(obs, dict) else str(obs)
    
    return {
        "observation": {"message": message, "raw": obs},
        "reward": {"alignment": float(reward)},
        "done": done,
        "info": info,
    }

@app.get("/state")
def state(session_id: str = "default"):
    return get_env(session_id).state()
