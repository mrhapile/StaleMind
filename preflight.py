import requests
import os

print("--- 1. Space Health ---")
try:
    r = requests.get("https://mrhapile-stalemind.hf.space")
    print("Status code:", r.status_code)
except Exception as e:
    print("Failed to connect:", e)

print("\n--- 2. Reward Format ---")
BASE_URL = "https://mrhapile-stalemind.hf.space"
try:
    r = requests.post(f"{BASE_URL}/reset", json={"scenario_index": 1})
    print("RESET:", r.json())

    r2 = requests.post(f"{BASE_URL}/step", json={"type": "ACCEPT", "content": ""})
    print("STEP:", r2.json())
    
    result = r2.json()
    if "reward" in result:
        reward = result["reward"]["score"] if isinstance(result["reward"], dict) else result["reward"]
        print("Detected Reward:", reward)
except Exception as e:
    print("Failed to connect for /reset and /step:", e)

print("\n--- 3. HF Token + Model Access ---")
try:
    from huggingface_hub import InferenceClient
    client = InferenceClient(
        model="meta-llama/Llama-3.2-3B-Instruct",
        token=os.getenv("HF_TOKEN")
    )
    print("Llama Output:", client.text_generation("Reply with: hello", max_new_tokens=10))
except Exception as e:
    print("HF Test failed:", e)
