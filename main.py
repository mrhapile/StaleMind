from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from env.environment import StaleMindEnv

app = FastAPI(title="StaleMind Environment API")

env = StaleMindEnv()
env.reset()

class ActionRequest(BaseModel):
    type: str
    content: Optional[str] = ""

class ResetRequest(BaseModel):
    scenario_index: Optional[int] = None

@app.post("/reset")
def reset(req: ResetRequest = None):
    index = req.scenario_index if req else None
    obs = env.reset(scenario_index=index)
    return {"observation": obs}

@app.post("/step")
def step(action: ActionRequest):
    act_dict = {"type": action.type, "content": action.content}
    obs, reward, done, info = env.step(act_dict)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return {"observation": env.state()}
