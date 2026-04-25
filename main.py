from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from env.environment import StaleMindEnv as DriftGym

app = FastAPI()

# Session storage
envs: Dict[str, DriftGym] = {}

def get_env(session_id: str) -> DriftGym:
    if session_id not in envs:
        envs[session_id] = DriftGym()
        envs[session_id].reset()
    return envs[session_id]

@app.get("/")
def home():
    return {"message": "StaleMind API running"}

class ResetRequest(BaseModel):
    scenario_index: Optional[int] = None
    session_id: str = "default"

class StepRequest(BaseModel):
    type: str
    content: str = ""
    session_id: str = "default"

@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    env = get_env(req.session_id)
    obs, _ = env.reset(req.scenario_index)
    return {"observation": obs}

@app.post("/step")
def step(action: StepRequest):
    env = get_env(action.session_id)
    obs, reward, done, _ = env.step(action.model_dump() if hasattr(action, 'model_dump') else action.dict())
    
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
        "done": done
    }

@app.get("/state")
def state(session_id: str = "default"):
    return get_env(session_id).state()
