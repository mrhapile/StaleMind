from fastapi import FastAPI
import gradio as gr
import uvicorn
import requests
import threading
import time

api = FastAPI()
@api.get("/hello")
def hello(): return {"msg": "hi"}

with gr.Blocks() as demo:
    gr.Markdown("Hello")

app = gr.mount_gradio_app(api, demo, path="/")

def run():
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="error")

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(1)
print("Root:", requests.get("http://127.0.0.1:8008/").text)
print("Hello:", requests.get("http://127.0.0.1:8008/hello").text)
print("Gradio:", requests.get("http://127.0.0.1:8008/gradio").text)
