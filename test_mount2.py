from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import uvicorn
import requests
import threading
import time

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@api.get("/hello")
def hello(): return {"msg": "hi"}

with gr.Blocks() as demo:
    gr.Markdown("Hello")

app = gr.mount_gradio_app(api, demo, path="/")

def run():
    uvicorn.run(app, host="127.0.0.1", port=8009, log_level="error")

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(1)
print("Root:", requests.get("http://127.0.0.1:8009/").text)
print("Hello:", requests.get("http://127.0.0.1:8009/hello").text)
