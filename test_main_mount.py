from main import app as api
import gradio as gr
import uvicorn
import requests
import threading
import time

with gr.Blocks() as demo:
    gr.Markdown("Hello")

app = gr.mount_gradio_app(api, demo, path="/")

def run():
    uvicorn.run(app, host="127.0.0.1", port=8010, log_level="error")

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(1)
print("Root:", requests.get("http://127.0.0.1:8010/").text)
